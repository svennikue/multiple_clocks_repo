#!/bin/bash
# Loop over numbers 1 to 57
for i in {1..57}; do

  # Skip the subjects we want to exclude
  if [[ "$i" == "9" || "$i" == "27" || "$i" == "43" || "$i" == "44" ]]; then
      continue
  fi
  
  # Print the subject (optional, for debugging)
  echo "Processing subject $i"
  
  # Call the Python script with the appropriate arguments
  python wrapper_human_cells_elnetreg.py "$i" \
    --models_I_want="['only','onlynowand3future', 'onlynextand2future']" \
    --exclude_x_repeats="[1,2,3]" \
    --randomised_reward_locations=False \
    --save_regs=True
done
