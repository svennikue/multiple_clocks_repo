scratchDir="/home/fs0/xpsy1114/scratch/data/derivatives"
analysisDir="/home/fs0/xpsy1114/scratch/analysis"


echo Scratch directory is $scratchDir
version="all-paths-fixed_stickrews_split-buttons"
echo this is version $version

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
# # for subjectTag in 02 ; do
# 	funcDir=$scratchDir/sub-$subjectTag/func
# 	for th in 1 2 ; do
# 		glmDir="$funcDir/glm_${version}_pt0${th}.feat"
# 		echo now checking $glmDir
# 		if [ -f $glmDir/stats/pe1.nii.gz ]; then
# 			echo glm ran for sub${subjectTag} and task half $th
# 		else
# 			echo CAREFUL! glm $version $th for sub$subjectTag FAILED! RUN AGAIN!
# 			echo
# 			#echo now removing $glmDir
# 			#rm -r $glmDir
# 		fi
#  	done
# done
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
#     anatDir="${scratchDir}/sub-${subjectTag}/anat"
# 	echo now checking $anatDir
# 	if [ -f $anatDir/grey_matter_mask_func_01.nii.gz ]; then
# 		echo the gm mask one exists for $subjectTag
# 	else
# 		echo the crucial file has not run yet!!! wait for this one!!!
# 	fi
# 	if [ -f $anatDir/grey_matter_mask_func_02.nii.gz ]; then
# 		echo the gm mask two exists for $subjectTag
# 	else
# 		echo the crucial file has not run yet!!! wait for this one!!!
# 	fi
# done


# /home/fs0/xpsy1114/scratch/data/derivatives/sub-31/func/RSA_DSR_bias-path-rew-splitfuts_combos_31-01-2026_glmbase_all-paths-fixed_stickrews_split-buttons/results/BUTTONS_OUT-DSR_vis_button_out_pathrew-reward-path_t_val.nii.gz
for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
	# for subjectTag in 30 31 32 ; do
	# data/derivatives/sub-01/func/RSA_state_action_playaround_19-02-2026_glmbase_all-paths-fixed_stickrews_split-buttons/results/DSR-DSR_stateaction_now_controls-mask_reward-path_beta.nii.gz
	# data/derivatives/sub-01/func/RSA_which-fut-isin-DSR_27-02-2026_glmbase_all-paths-fixed_stickrews_split-buttons/standard-space-smooth/smooth_fwhm5_NEXT3_QUARTER-split_quarters_DSR-mask_reward-path_t_val_std.nii.gz

	candidates=( ${scratchDir}/sub-${subjectTag}/func/RSA_quarters_DSR_controls_*_glmbase_all-paths-fixed_stickrews_split-buttons/standard-space-smooth )
	# candidates=( ${scratchDir}/sub-${subjectTag}/func/RSA_fixed_DSR-physabst-state_*_glmbase_all-paths-fixed_stickrews_split-buttons/results )
    # echo $candidates
	if ((${#candidates[@]})); then
        # If multiple dates exist, pick the newest by mtime
        IFS=$'\n' candidates=($(ls -1dt "${candidates[@]}"))
        funcDir="${candidates[0]}"
	fi

	echo now checking $funcDir

	# if [ -f $funcDir/smooth_fwhm5_action_DSR_beta_std.nii.gz ]; then
	# #if [ -f $funcDir/A-STATE-state_controls-reward-reward_t_val.nii.gz ]; then
	# 	echo the crucial state file exists for $subjectTag
	# else
	# 	echo the crucial file has not run yet!!! wait for this one!!!
	# fi

	if [ -f $funcDir/smooth_fwhm5_ROT_NEXT3_QUARTER-split_rot_quarters_DSR-mask_reward-path_t_val_std.nii.gz ]; then
		echo and the smoothed file also already exists.
		echo
	else
		echo the smoothed file has not run yet!!! wait for $subjectTag !!!
	fi
done
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35; do
# 	oldRDMdir=$scratchDir/sub-$subjectTag/func/data_RDMs_DSR_rew-vs-path_interact_vis_combos_27-01-2026_glmbase_all-paths-fixed_stickrews_split-buttons
# 	echo now renaming $oldRDMdir
# 	mv $oldRDMdir $scratchDir/sub-$subjectTag/func/data_RDMs_glmbase_all-paths-fixed_stickrews_split-buttons
# done
# # # data/derivatives/sub-03/func/RSA_state_and_combo_11-12-2025_glmbase_all_paths-stickrews-split_buttons/
# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
# # for subjectTag in 02 ; do
# # data/derivatives/sub-01/func/RSA_DSR_stepwise_combos_14-01-2026_glmbase_all_paths-stickrews-split_buttons
# 	funcDir=$scratchDir/sub-$subjectTag/func/RSA_DSR_rew-vs-path_stepwise_combos_20-01-2026_glmbase_all-paths-fixed_stickrews_split-buttons/results/
# 	# echo now checking $funcDir
# 	# if [ -f $funcDir/results/STATE-state_stateA_locs_l2_t_val.nii.gz ]; then
# 	# 	echo the crucial state file exists for $subjectTag
# 	# else
# 	# 	echo the crucial file has not run yet!!! wait for this one!!!
# 	# fi
# 	if [ -f $funcDir/A-STATE-DSR_state_locs_l2_Astate_p_val.nii.gz ]; then
# 		echo most likely, all results file exist
# 		echo
# 	else
# 		echo the RSA has not run yet!!! double check for $subjectTag !!!
# 	fi
# done

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
# # for subjectTag in 02 ; do
# 	funcDir=$scratchDir/sub-$subjectTag/func
# 	for th in 1 2 ; do
# 		glmDir="$funcDir/glm_${version}_pt0${th}.feat"
# 		echo now checking $glmDir
# 		if [ -f $glmDir/stats/pe1.nii.gz ]; then
# 			echo glm for sub${subjectTag} and task half $th exists, all good
# 		else
# 			echo CAREFUL!!!!
# 			echo 
# 			echo $glmDir for sub${subjectTag} and task half $th DOESNT EXIST
# 			echo check this
# 			#echo now removing $glmDir
# 			#rm -r $glmDir
# 		fi
# 	done
# done

# echo Scratch directory is $scratchDir
# version="02-l"
# echo this is version $version

# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
# # for subjectTag in 02 ; do
# 	funcDir=$scratchDir/sub-$subjectTag/func
# 	for th in 1 2 ; do
# 		glmDir="$funcDir/glm_${version}_pt0${th}.feat"
# 		#echo now checking $glmDir
# 		echo now removing $glmDir
# 		rm -r $glmDir
# 	done
# done


# for subjectTag in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 22 23 24 25 26 27 28 30 31 32 33 34 35; do
# # for subjectTag in 01 02 ; do
# 	funcDir=$scratchDir/sub-$subjectTag/func
# 	for th in 1 2 ; do
# 		glmDir="$funcDir/glm_${version}_pt0${th}.feat"
# 		echo now removing $glmDir
# 		rm -r $glmDir
# 	done
# done
