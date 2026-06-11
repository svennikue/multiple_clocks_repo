#!/bin/bash
set -euo pipefail

scratchDir="/home/fs0/xpsy1114/scratch/data"
module load fsl || true

glm_version="all-paths-fixed_stickrews_split-buttons"
RSA_version="DSR_quarters_except_prev_button_state"

groupDir="${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}_cropped"
outDir="${groupDir}_masked"

mask="${scratchDir}/masks/PFC/mask_PFC_LR_smoothed_resampled.nii.gz"

mkdir -p "$outDir"

mask_file() {
  local file="$1"
  local name base out

  name=$(basename "$file")

  # remove .nii or .nii.gz
  base="${name%.nii.gz}"
  base="${base%.nii}"

  # make the filename shorter
  base="${base#cropped_masked_smooth_fwhm5_}"
  base="${base#cropped_smooth_fwhm5_}"
  base="${base%-mask_reward-path_beta_std}"

  out="${outDir}/${base}_masked.nii.gz"

  echo "Masking $name"
  fslmaths "$file" -mul "$mask" "$out"
}

export -f mask_file
export outDir mask

find "$groupDir" -maxdepth 1 -type f \( \
  -name "*hm5_ROT_*.nii" -o \
  -name "*hm5_ROT_*.nii.gz" -o \
  -name "*QUARTER-split*.nii" -o \
  -name "*QUARTER-split*.nii.gz" -o \
  -name "*FUT-split_eighths*.nii" -o \
  -name "*FUT-split_eighths*.nii.gz" \
\) -print0 | while IFS= read -r -d '' file; do
  mask_file "$file"
done

echo "Done. Masked files are in:"
echo "$outDir"