dataDirServer="/ceph/behrens/svenja/human_ABCD_ephys"

for i in $(seq -w 1 50); do
  subjects+=("$i")
done

# subjects=("1" "2")

for sub in "${subjects[@]}"; do
    # source_path="$laptopDir/s$sub"
    mkdir -p "$dataDirServer/s$sub"
    mkdir -p "$dataDirServer/s$sub/anat"
    mkdir -p "$dataDirServer/s$sub/LFP"
done