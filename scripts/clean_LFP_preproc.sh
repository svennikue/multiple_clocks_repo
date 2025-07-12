#!/bin/bash

# Base directory path
BASE_DIR="/ceph/behrens/svenja/human_ABCD_ephys/derivatives"

echo "Starting cleanup of LFP folders..."

# Loop through s01 to s59
for i in $(seq -w 1 59); do
    LFP_DIR="$BASE_DIR/s$i/LFP"
    
    if [ -d "$LFP_DIR" ]; then
        echo "Emptying $LFP_DIR"
        rm -f "$LFP_DIR"/* 2>/dev/null
    else
        echo "Skipping s$i — no LFP folder found."
    fi
done

echo "Done cleaning all LFP folders."
