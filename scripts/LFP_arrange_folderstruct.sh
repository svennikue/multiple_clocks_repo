scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans"
derivDir="${scratchDir}/derivatives"
if [ ! -d "$derivDir" ]; then
    mkdir $derivDir
    echo created $derivDir
fi

for i in $(seq -w 1 50); do
  subjects+=("$i")
done

for sub in "${subjects[@]}"; do
    subDerivDir=$derivDir/s$sub
    if [ ! -d "$subDerivDir" ]; then
        mkdir $subDerivDir
        mkdir -p "$subDerivDir/anat"
        mkdir -p "$subDerivDir/LFP"
        echo created all sub dirs for $subDerivDir
    fi
done