scratchDir="/home/fs0/xpsy1114/scratch/data"

for d in $scratchDir/data/derivatives/sub-*/func/glm*; do
    [ -d "$d" ] || continue
    n=$(find "$d" -maxdepth 1 -type f | wc -l)
    if ["$n" -lt 10 ]; then echo "<10 $n $d"; fi
    case "$d" in *glm_fut*|"glm_all") if ["$n" -gt 10]; then echo ">10 $n $d"; fii;; esac
done
