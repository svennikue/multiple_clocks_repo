# multiple_clocks_repo
This repo includes 
1. the package 'mc' which allows to run task-configuration and solving simulations, create models of different neurons, and do some computations with these models.
2. scripts that I use to run this package, which 

/a) runs the simulation with the following selection of settings:
Create the task, Create a distribution of most common pathlengths, Setting the Clocks and Location Matrix, Setting the Clocks + locs but in 'real time' + HRF convolve, Setting 0-phase clocks in 'real time', concatenate 400 HRF convolved clocks and PCA, Create 'neuron plots', Create RDMs, create RDMS between 0 phase clock and clocks (HRF + by time)
  
/b) go beyond running the simulation and creating predicitons, but actually prepare the MRI experiment by optimizing for certain things (loop_through_tasks.py), with the following options:
settings for the rest of the script
find_low_similarity_within = 0
find_low_similarity_between = 0
find_low_sim_zerophase_clocks = 0
distr_zero_phase_clocks_optimal = 1
plot_optimal_paths = 0
run_PCA_on_repetitions = 0


  
 
