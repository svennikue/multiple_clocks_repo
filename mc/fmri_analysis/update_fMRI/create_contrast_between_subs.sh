#!/bin/bash

# Set scratch directory for execution on server
scratchDir="/home/fs0/xpsy1114/scratch/data"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"

# RSA_version="DSR_hamming_split_comps_combos"
glm_version="all-paths-fixed_stickrews_split-buttons"
RSA_version="which-fut-isin-DSR"

fslDir="/opt/fmrib/fsl"
export fslDir=~/scratch/fsl
export PATH=$fslDir/share/fsl/bin/:$PATH
source $fslDir/etc/fslconf/fsl.sh
module load fsl


groupDir=${scratchDir}/derivatives/group/group_RSA_${RSA_version}_glmbase_${glm_version}_cropped
contrastDir=${scratchDir}/derivatives/group/contrast_RSA_${RSA_version}_glmbase_${glm_version} 


if [ ! -d $contrastDir ]; then 
    mkdir $contrastDir 
fi

echo this is group dir $groupDir


# prefixes used in your filenames
PREFIX_one="cropped_masked_smooth_fwhm5_CURR_"
PREFIX_two="cropped_masked_smooth_fwhm5_NEXT_"
PREFIX_three="cropped_masked_smooth_fwhm5_NEXT2_"
PREFIX_four="cropped_masked_smooth_fwhm5_NEXT3_"

# the 4 variant stems you want processed (the first is the vis-mot-controls one you used before,
# the other three are the "allcontrols" variants)

SUFFIX="QUARTER-split_quarters_DSR-mask_reward-path_beta_std.nii"


# create contrasts for the rotations

in1="${groupDir}/${PREFIX_one}${SUFFIX}"
in2="${groupDir}/${PREFIX_two}${SUFFIX}"
in3="${groupDir}/${PREFIX_three}${SUFFIX}"
in4="${groupDir}/${PREFIX_four}${SUFFIX}"
out="${contrastDir}/quarters_DSR_contrast_1_1_mean.nii.gz"

if [ ! -f "$in1" ]; then
  echo "ERROR: missing input $in1" >&2
  continue
fi
if [ ! -f "$in2" ]; then
  echo "ERROR: missing input $in2" >&2
  continue
fi
if [ ! -f "$in3" ]; then
  echo "ERROR: missing input $in3" >&2
  continue
fi
if [ ! -f "$in4" ]; then
  echo "ERROR: missing input $in4" >&2
  continue
fi

# ensure same number of 4th-dim volumes
nvol1=$(fslval "$in1" dim4 2>/dev/null || echo 1)
nvol2=$(fslval "$in2" dim4 2>/dev/null || echo 1)
nvol3=$(fslval "$in3" dim4 2>/dev/null || echo 1)
nvol4=$(fslval "$in4" dim4 2>/dev/null || echo 1)
if [ "$nvol1" != "$nvol2" ] || [ "$nvol1" != "$nvol3" ] || [ "$nvol1" != "$nvol4" ]; then
  echo "ERROR: volume count mismatch for $in1 ($nvol1), $in2 ($nvol2), $in3 ($nvol3), or $in4 ($nvol4)" >&2
  continue
fi

echo "Computing (quarter1+2+3+4)/4 for $SUFFIX"
fslmaths "$in1" -add "$in2" -add "$in3" -add "$in4" -div 4 "$out"
echo "Wrote $out"


echo done!



# # prefixes used in your filenames
# DSR_PREFIX="masked_smooth_fwhm5_DSR-DSR_stateaction_"
# STATE_PREFIX="masked_smooth_fwhm5_STATE_ACTION_GLOB-DSR_stateaction_"

# # the 4 variant stems you want processed (the first is the vis-mot-controls one you used before,
# # the other three are the "allcontrols" variants)

# variants=(
#   "vis-mot-controls-mask_reward-path_beta_std.nii"
#   "vis-mot-controls-path-path_beta_std.nii"
#   "vis-mot-controls-reward-reward_beta_std.nii"
#   "allcontrols-mask_reward-path_beta_std.nii"
#   "allcontrols-path-path_beta_std.nii"
#   "allcontrols-reward-reward_beta_std.nii"
# )

# # create contrasts for DSR vs STATE for each variant
# for v in "${variants[@]}"; do
#   in1="${groupDir}/${DSR_PREFIX}${v}"
#   in2="${groupDir}/${STATE_PREFIX}${v}"
#   out="${contrastDir}/contrast_1_1_mean_${v%.nii}.nii.gz"

#   if [ ! -f "$in1" ]; then
#     echo "ERROR: missing input $in1" >&2
#     continue
#   fi
#   if [ ! -f "$in2" ]; then
#     echo "ERROR: missing input $in2" >&2
#     continue
#   fi

#   # ensure same number of 4th-dim volumes
#   nvol1=$(fslval "$in1" dim4 2>/dev/null || echo 1)
#   nvol2=$(fslval "$in2" dim4 2>/dev/null || echo 1)
#   if [ "$nvol1" != "$nvol2" ]; then
#     echo "ERROR: volume count mismatch for $in1 ($nvol1) and $in2 ($nvol2)" >&2
#     continue
#   fi

#   echo "Computing (DSR + STATE)/2 for variant: $v"
#   fslmaths "$in1" -add "$in2" -div 2 "$out"
#   echo "Wrote $out"
# done

# echo done!
