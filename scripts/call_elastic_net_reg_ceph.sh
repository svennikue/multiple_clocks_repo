#!/bin/bash
# Loop over numbers 1 to 57
for i in {1..59}; do

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
    --comp_loc_perms=1000

done

