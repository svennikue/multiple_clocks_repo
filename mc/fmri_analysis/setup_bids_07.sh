#this script sets up the BIDS structure on the cluster and renames all files
#for the first pilot scan 

scratchDir="/home/fs0/xpsy1114/scratch/data"
# rawDir="/home/fs0/xpsy1114/scratch/data/raw/01_pilot"
# subjectTag="02"
# submit like: bash setup_bids.sh "/home/fs0/xpsy1114/scratch/data/raw/07_scan" 07
rawDir="$1"
subjectTag="$2"

# set up BIDs folder structure for raw data.
mkdir -p $scratchDir/pilot
mkdir -p $scratchDir/pilot/sub-$subjectTag

# Construct structural directory for raw file
anatDir=$scratchDir/pilot/sub-$subjectTag/anat
# And create directory for derived anatomy files
mkdir -p $anatDir

# Construct functional directory for raw file
funcDir=$scratchDir/pilot/sub-$subjectTag/func
# And create directory for derived functiona; files
mkdir -p $funcDir

# Construct fieldmap directory for raw file
fmapDir=$scratchDir/pilot/sub-$subjectTag/fmap
# And create directory for derived fieldmap files
mkdir -p $fmapDir

# Construct behavioural directory for raw file
behDir=$scratchDir/pilot/sub-$subjectTag/beh
# And create directory for derived behavioural files
mkdir -p $behDir

# Construct physiology directory for raw file
motionDir=$scratchDir/pilot/sub-$subjectTag/motion
# And create directory for derived motion files
mkdir -p $motionDir


# now copy and rename files

# anat
cp $rawDir/*_MPRAGE_UP.nii $anatDir/
cp $anatDir/*_MPRAGE_UP.nii $anatDir/sub-${subjectTag}_T1w.nii
gzip $anatDir/sub-${subjectTag}_T1w.nii
rm $anatDir/sub-${subjectTag}_T1w.nii

#func
# single volume functional 1
cp $rawDir/images_06*.nii $funcDir/
cp $funcDir/images_06*.nii $funcDir/sub-${subjectTag}_1_vol_1_bold.nii
gzip $funcDir/sub-${subjectTag}_1_vol_1_bold.nii
rm $funcDir/sub-${subjectTag}_1_vol_1_bold.nii

# func 1 
cp $rawDir/images_07*.nii $funcDir/
cp $funcDir/images_07*.nii $funcDir/sub-${subjectTag}_1_bold.nii
gzip $funcDir/sub-${subjectTag}_1_bold.nii
rm $funcDir/sub-${subjectTag}_1_bold.nii

# single volume functional 2
cp $rawDir/images_09*.nii $funcDir/
cp $funcDir/images_09*.nii $funcDir/sub-${subjectTag}_1_vol_2_bold.nii
gzip $funcDir/sub-${subjectTag}_1_vol_2_bold.nii
rm $funcDir/sub-${subjectTag}_1_vol_2_bold.nii

# func 2 
cp $rawDir/images_010*.nii $funcDir/
cp $funcDir/images_010*.nii $funcDir/sub-${subjectTag}_2_bold.nii
gzip $funcDir/sub-${subjectTag}_2_bold.nii
rm $funcDir/sub-${subjectTag}_2_bold.nii

# func whole volume
cp $rawDir/images_011*.nii $funcDir/
cp $funcDir/images_011*.nii $funcDir/sub-${subjectTag}_1_bold_wb.nii
gzip $funcDir/sub-${subjectTag}_1_bold_wb.nii
rm $funcDir/sub-${subjectTag}_1_bold_wb.nii

cp $rawDir/images_012*.nii $funcDir/
cp $funcDir/images_012*.nii $funcDir/sub-${subjectTag}_2_bold_wb.nii
gzip $funcDir/sub-${subjectTag}_2_bold_wb.nii
rm $funcDir/sub-${subjectTag}_2_bold_wb.nii

 
#fmap
cp $rawDir/images*field*e1.json $fmapDir/
cp $fmapDir/images*field*e1.json $fmapDir/sub-${subjectTag}_magnitude1.json
cp $rawDir/images*field*e1.nii $fmapDir/
cp $fmapDir/images*field*e1.nii $fmapDir/sub-${subjectTag}_magnitude1.nii
gzip $fmapDir/sub-${subjectTag}_magnitude1.nii
rm $fmapDir/sub-${subjectTag}_magnitude1.nii

cp $rawDir/images*field*e2.json $fmapDir/
cp $fmapDir/images*field*e1.json $fmapDir/sub-${subjectTag}_magnitude2.json
cp $rawDir/images*field*e2.nii $fmapDir/
cp $fmapDir/images*field*e2.nii $fmapDir/sub-${subjectTag}_magnitude2.nii
gzip $fmapDir/sub-${subjectTag}_magnitude2.nii
rm $fmapDir/sub-${subjectTag}_magnitude2.nii

cp $rawDir/images*field*e2_ph.json $fmapDir/
cp $fmapDir/images*field*e2_ph.json $fmapDir/sub-${subjectTag}_phasediff.json
cp $rawDir/images*field*e2_ph.nii $fmapDir/
cp $fmapDir/images*field*e2_ph.nii $fmapDir/sub-${subjectTag}_phasediff.nii
gzip $fmapDir/sub-${subjectTag}_phasediff.nii
rm $fmapDir/sub-${subjectTag}_phasediff.nii


# check if I have these files/ whether I can get them!!

# behav

# motion

