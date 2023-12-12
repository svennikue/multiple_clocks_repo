# multiple_clocks_repo
This repo includes 
1. the package 'mc' which allows to run task-configuration and solving simulations, create models of different neurons, and do some computations with these models, as well as analyse/preprocess MRI data
2. scripts which use functions of this package, that
   a) run the simulation with the following selection of settings: Create the task, Create a distribution of most common pathlengths, Setting the Clocks and Location Matrix, Setting the Clocks + locs but in 'real time' + HRF convolve, Setting 0-phase clocks in 'real time', concatenate 400 HRF convolved clocks and PCA, Create 'neuron plots', Create RDMs, create RDMS between 0 phase clock and clocks (HRF + by time)
  b) go beyond running the simulation and creating predicitons, but actually prepare the MRI experiment by optimizing for certain things (loop_through_tasks.py)
  c) Analyse rodent data (El-Gaby 2023) with RSA and validating the model simulations
  d) create EVs for first-level FEAT based on behavioural data to generate RDM conditions
  e) create model RDMs for fMRI analysis based on behavioural data
  f) run RSA with fMRI data and model RDMs


  
 
