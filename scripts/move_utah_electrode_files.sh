#!/bin/bash

# Set source and destination directories
SRC_DIR="/ceph/behrens/svenja/ABCD_LFP"
DEST_DIR="/ceph/behrens/svenja/human_ABCD_ephys"

echo "Starting file move process..."



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
    mkdir -p "$DEST_DIR/$newbase/anat"
    mkdir -p "$DEST_DIR/$newbase/electrodes"

    # move .ns2 files
    find "$folder" -maxdepth 1 -type f -name "*.ns2" -exec mv {} "$DEST_DIR/$newbase/LFP/" \;

    # move entire 'registered' subfolder
    if [ -d "$folder/Registered" ]; then
        mv "$folder/Registered/Electrodes.mat" "$DEST_DIR/$newbase/electrodes/Electrodes.mat"
        mv "$folder/Registered/CT.nii" "$DEST_DIR/$newbase/anat/CT.nii"
        mv "$folder/Registered/MR.nii" "$DEST_DIR/$newbase/anat/MRI.nii"
    fi

    echo "Finished processing $folder → $newbase"
done

echo "All done."



# #!/bin/bash

# BASE_DIR="/ceph/behrens/svenja/human_ABCD_ephys"


# # 3. Loop through s01 to s59 and delete subfolders
# for i in $(seq -w 1 59); do
#     SUBJECT_DIR="$BASE_DIR/s$i"
    
#     # 1. Delete the MR.nii file in s01/anat
#     rm -f "$SUBJECT_DIR/anat/MR.nii"

#     # 2. Move Electrode.mat to the correct path in s01
#     mkdir -p "$SUBJECT_DIR/electrodes"
#     mv "$SUBJECT_DIR/LFP/electrodes/Electrodes.mat" "$SUBJECT_DIR/electrodes/" 2>/dev/null


#     # Delete /electrodes and /LFP/anat if they exist
#     rm -rf "$SUBJECT_DIR/LFP/electrodes"
#     rm -rf "$SUBJECT_DIR/LFP/anat"

# done

# # 4. Delete misnamed directories like s1, s2, ..., s9, s10, etc., that are not s01-s59
# for folder in "$BASE_DIR"/s*; do
#     dirname=$(basename "$folder")
    
#     # Skip valid s01 to s59
#     if [[ "$dirname" =~ ^s[0-9]{2}$ ]]; then
#         continue
#     fi

#     # Delete all others like s1, s2, s123, etc.
#     rm -rf "$folder"
# done

# echo "Cleanup complete."
