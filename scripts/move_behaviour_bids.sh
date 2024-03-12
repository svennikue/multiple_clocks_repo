
# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

# If this is not called on the server, but on a laptop:
if [ ! -d $scratchDir ]; then
  scratchDir="/Users/xpsy1114/Documents/projects/multiple_clocks/data"
  analysisDir="/Users/xpsy1114/Documents/projects/multiple_clocks/multiple_clocks_repo/mc/fmri_analysis"
fi

# 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
for subjectTag in 27 28 29 30 31; do
    #mkdir ${scratchDir}/pilot/sub-${subjectTag}
    #mkdir ${scratchDir}/pilot/sub-${subjectTag}/beh
    #mkdir ${scratchDir}/pilot/sub-${subjectTag}/motion
    #cp ${scratchDir}/pilot/sub-${subjectTag}/motion/s_${subjectTag}*.txt ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.txt
    #cp ${scratchDir}/pilot/sub-${subjectTag}/motion/s_${subjectTag}*.acq ${scratchDir}/pilot/sub-${subjectTag}/motion/sub-${subjectTag}_physio.acq
    cp ${scratchDir}/pilot/sub-${subjectTag}/beh/*pt1*.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt1_all.csv
    cp ${scratchDir}/pilot/sub-${subjectTag}/beh/*pt2*.csv ${scratchDir}/pilot/sub-${subjectTag}/beh/sub-${subjectTag}_fmri_pt2_all.csv
done
