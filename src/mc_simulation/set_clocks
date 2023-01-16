import numpy as np


def set_clocks(walked_path, step_number, phases):
    # import pdb; pdb.set_trace()
    n_states = len(step_number)
    n_columns = phases*n_states
    # and number of rows is locations*phase*neurons per clock
    # every field (9 fields) -> can be the anchor of 3 phase clocks
    # -> of 12 neurons each. 9 x 3 x 12 
    # to make it easy, the number of neurons = number of one complete loop (12)
    n_rows = 9*phases*(phases*n_states)    
    clocks_matrix = np.empty([n_rows,n_columns]) # fields times phases.
    clocks_matrix[:] = np.nan
    phase_loop = list(range(0,phases))
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1]
    all_phases = list(range(n_columns))  
    # set up neurons of a clock.
    clock_neurons = np.zeros([phases*n_states,phases*n_states])
    for i in range(0,len(clock_neurons)):
        clock_neurons[i,i]= 1   
    # Now the big loop starts. Going through all subpathes, and interpolating
    # phases and steps.                  
    # if pathlength < phases:
    # it can be either pathlength == 1 or == 2. In both cases,
    # so dublicate the field until it matches length phases
    # if pathlength > phases:
    # dublicate the first phase so it matches length of path
    # so that, if finally, pathlength = phases
    # zip both lists and loop through them together.
    for count_paths, (pathlength) in enumerate(step_number):
        phasecount = len(phase_loop) #this needs to be reset for every subpath.
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        if pathlength < phasecount: 
            finished = False
            while not finished:
                curr_path.insert(0, curr_path[0]) # dublicate first field 
                pathlength = len(curr_path)
                finished = pathlength == phasecount
        elif pathlength > phasecount:
            finished = False
            while not finished:
                phase_loop.insert(0,phase_loop[0]) #make more early phases
                phasecount = len(phase_loop)
                finished = pathlength == phasecount
        if pathlength == phasecount:
            for phase, step in zip(phase_loop, curr_path):
                x = step[0]
                y = step[1]
                anchor_field = x + y*3
                anchor_phase_start = (anchor_field * n_columns * 3) + (phase * n_columns)
                initiate_at_phase = all_phases[phase+(count_paths*phases)]
                # check if clock has been initiated already
                is_initiated = clocks_matrix[anchor_phase_start, 0]
                if np.isnan(is_initiated): # if not initiated yet
                    # slice the clock neuron filler at the phase we are currently at.
                    first_split = clock_neurons[:, 0:(n_columns-initiate_at_phase)]
                    second_split = clock_neurons[:, n_columns-initiate_at_phase:None]
                    # then add another row of 1s and fill the matrix with it
                    fill_clock_neurons = np.concatenate((second_split, first_split), axis =1)
                    # and only the second part will be filled in.
                    clocks_matrix[anchor_phase_start:(anchor_phase_start+12), 0:None]= fill_clock_neurons
                else:
                    # if the clock has already been initiated and thus is NOT nan
                    # first take the already split clock from before within the whole matrix
                    # then identify the initiation point to fill in the new activation pattern
                    # (this is anchor_phase_start)
                    # now simply change the activity already in the matrix with a neuron-loop
                    for neuron in range(0, n_columns-initiate_at_phase):
                        # now the indices will be slightly more complicated...
                        # [row, column]
                        clocks_matrix[(anchor_phase_start + neuron), (neuron + initiate_at_phase)] = 1
                    if initiate_at_phase > 0:
                        for neuron in range(0, initiate_at_phase):
                            clocks_matrix[(anchor_phase_start + n_columns - initiate_at_phase + neuron), neuron] = 0  
    return clocks_matrix, total_steps  
    