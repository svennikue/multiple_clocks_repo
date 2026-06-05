#!/bin/bash
set -euo pipefail

scratchDir="/home/fs0/xpsy1114/scratch/data"

# keep your original FSL lines — if needed define fslDir before sourcing.
#source $fslDir/etc/fslconf/fsl.sh
module load fsl || true

glm_version="all-paths-fixed_stickrews_split-buttons"
RSA_version="quarters_DSR_controls"

groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}
croppedDir=${groupDir}_cropped
mkdir -p "$croppedDir"

# small threshold to detect "present" (non-zero) voxels
eps=1e-6

# grey matter mask
gm_mask=$scratchDir/masks/gm_mask_resampled.nii.gz 

# -----------------------
# helper: presence-count via per-volume loop (used only once)
compute_presence_by_loop() {
  local input4D="$1"
  local presence_out="$2"
  local nvol="$3"
  local tmpd
  tmpd=$(mktemp -d "${croppedDir}/tmp_presence.XXXX") || return 1

  # initialize accumulator to zeros using first volume header
  fslroi "$input4D" "${tmpd}/vol0.nii.gz" 0 1
  fslmaths "${tmpd}/vol0.nii.gz" -mul 0 "${tmpd}/acc.nii.gz"

  for ((i=0;i<nvol;i++)); do
    vol="${tmpd}/vol_$(printf "%03d" $i).nii.gz"
    binvol="${tmpd}/bin_$(printf "%03d" $i).nii.gz"
    fslroi "$input4D" "$vol" $i 1
    fslmaths "$vol" -abs -thr "$eps" -bin "$binvol"
    fslmaths "${tmpd}/acc.nii.gz" -add "$binvol" "${tmpd}/acc.nii.gz"
  done

  mv "${tmpd}/acc.nii.gz" "$presence_out"
  rm -rf "$tmpd"
  return 0
}
# -----------------------

# Find a representative 4D file to compute the shared mask (first regular file)
rep_file=""
for f in "$groupDir"/*; do
  [ -f "$f" ] || continue
  rep_file="$f"
  break
done

if [ -z "$rep_file" ]; then
  echo "ERROR: no files found in $groupDir" >&2
  exit 1
fi

echo "Using representative file to build mask: $(basename "$rep_file")"

# Read/clean dim4 and require 33 subjects
nvol_raw=$(fslval "$rep_file" dim4 2>/dev/null || true)
nvol=$(echo "${nvol_raw}" | tr -cd '0-9')
if [ -z "$nvol" ]; then
  echo "ERROR: couldn't read numeric dim4 from $rep_file (got: '$nvol_raw')" >&2
  exit 1
fi
echo "Detected $nvol volumes in rep file"

required_n=33
if [ "$nvol" -ne "$required_n" ]; then
  echo "ERROR: rep file has $nvol volumes but you requested mask for $required_n datapoints." >&2
  exit 1
fi

# create presence and mask (one-time)
rep_base="${rep_file##*/}"
rep_base="${rep_base%%.*}"
presence="${croppedDir}/${rep_base}_presence_count.nii.gz"
mask="${croppedDir}/mask_all_${required_n}_subjects.nii.gz"

echo "Computing presence-count (one-time) -> $presence"
if ! compute_presence_by_loop "$rep_file" "$presence" "$nvol"; then
  echo "ERROR: failed to compute presence for $rep_file" >&2
  exit 1
fi

echo "Creating mask of voxels present in all ${required_n} subjects -> $mask"
if ! fslmaths "$presence" -thr "$required_n" -uthr "$required_n" -bin "$mask"; then
  echo "ERROR: fslmaths failed creating mask from $presence" >&2
  exit 1
fi

# additionally restrict to grey matter mask
if ! fslmaths "$mask" -mul "$gm_mask" "$mask"; then
  echo "ERROR: failed to apply grey matter mask" >&2
  exit 1
fi

nvox_total=$(fslstats "$mask" -V 2>/dev/null | awk '{print $1}')
echo "Mask keeps $nvox_total voxels present in all $required_n subjects."
if [ "$nvox_total" -eq 0 ]; then
  echo "ERROR: resulting mask is empty. Aborting." >&2
  exit 1
fi

# Now apply this same mask to every file in groupDir
for file in "$groupDir"/*; do
  [ -f "$file" ] || continue
  filename=$(basename "$file")
  out4D="${croppedDir}/cropped_${filename}"
  echo "Applying mask to $filename -> $(basename "$out4D")"
  if ! fslmaths "$file" -mul "$mask" "$out4D"; then
    echo "WARNING: failed to mask $file; skipping." >&2
    continue
  fi
  echo " Wrote $out4D"
done

echo "All done. One mask was computed and applied to all files."
