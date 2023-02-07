# multiple_clocks_repo
This repo includes 
1. the package 'mc' which allows to run task-configuration and solving simulations, create models of different neurons, and do some computations with these models.
2. scripts that I use to run this package, which 

a) runs the simulation with the following selection of settings:
section_oneone = 1 # Create the task
section_onetwo = 0 # Create a distribution of most common pathlengths
section_twoone = 0 # Setting the Clocks and Location Matrix. 
section_twotwo = 1 # Setting the Clocks + locs but in 'real time' + HRF convolve
section_twothree = 0 # Setting 0-phase clocks in 'real time'
section_twofour = 0 # concatenate 400 HRF convolved clocks and PCA
section_three = 0 # Create 'neuron plots'
section_fourone = 0 # Create RDMs.
section_fourtwo = 0 # create RDMS between 0 phase clock and clocks (HRF + by time)
  
b) go beyond running the simulation and creating predicitons, but actually prepare the MRI experiment by optimizing for certain things (loop_through_tasks.py), with the following options:
settings for the rest of the script
find_low_similarity_within = 0
find_low_similarity_between = 0
find_low_sim_zerophase_clocks = 0
distr_zero_phase_clocks_optimal = 1
plot_optimal_paths = 0
run_PCA_on_repetitions = 0


  
 
