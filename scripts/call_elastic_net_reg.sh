#!/bin/bash
# Loop over numbers 1 to 57
# for i in {1..59}; do
for i in {1..11}; do

  # Skip the subjects we want to exclude
  # 27 and 44 have something wrong with location/reward timing
  if [[ "$i" == "27" || "$i" == "44" ]]; then
      continue
  fi
  
  # Print the subject (optional, for debugging)
  echo "Processing subject $i"
  
  # Call the Python script with the appropriate arguments
  python wrapper_human_cells_elnetreg.py "$i" \
    --models_I_want="['onlynowand3future', 'onlynextand2future']" \
    --exclude_x_repeats="[1]" \
    --randomised_reward_locations=False \
    --save_regs=True \
    --fit_binned='by_state' \
    --fit_residuals=False \
    --avg_across_runs=True \
    --comp_time_perms=300

  #    python wrapper_human_cells_elnetreg.py "$i" \
  #  --models_I_want="['only','onlynowand3future','onlynextand2future']" \
  #  --exclude_x_repeats="[1,2,3]" \
  #  --randomised_reward_locations=False \
  #  --save_regs=True \
  #  --bin_pre_corr='by_state_loc_change' \
  #  --avg_across_runs=True
    
done
