import numpy as np

def set_location_matrix(walked_path, step_number, phases):
    #import pdb; pdb.set_trace()
    n_states = len(step_number)
    n_columns = phases*n_states
    location_matrix = np.zeros([9,n_columns]) # fields times phases.
    phase_loop = list(range(0,phases))
    cumsumsteps = np.cumsum(step_number)
    total_steps = cumsumsteps[-1] # DOUBLE CHECK IF THIS IS TRUE
    # I will use the same logic as with the clocks. The first step is to take
    # each subpath isolated, since the phase-neurons are aligned with the phases (ie. reward)
    # then, I check if the pathlength is the same as the phase length.
    # if not, I will adjust either length, and then use the zip function 
    # to loop through both together and fill the matrix.
    for count_paths, (pathlength) in enumerate(step_number):
        print('Entered loop which goes through every subpath, currently at', count_paths)
        phasecount = len(phase_loop) #this needs to be reset for every subpath.
        if count_paths > 0:
            curr_path = walked_path[cumsumsteps[count_paths-1]+1:(cumsumsteps[count_paths]+1)]
        elif count_paths == 0:
            curr_path = walked_path[1:cumsumsteps[count_paths]+1]
        print('Now I defined the current walked path:', curr_path)
        # if pathlength < phases -> 
        # it can be either pathlength == 1 or == 2. In both cases,
        # dublicate the field until it matches length phases
        # if pathlength > phases
        # dublicate the first phase so it matches length of path
        # so that, if finally, pathlength = phases
        # zip both lists and loop through them together.
        if pathlength < phasecount: 
            finished = False
            print('Entered a loop for paths shorter than 3 phases')
            while not finished:
                curr_path.insert(0, curr_path[0]) # dublicate first field 
                pathlength = len(curr_path)
                finished = pathlength == phasecount
        elif pathlength > phasecount:
            finished = False
            print('Entered a loop for paths longer than 3 phases')
            while not finished:
                phase_loop.insert(0,phase_loop[0]) #make more early phases
                phasecount = len(phase_loop)
                finished = pathlength == phasecount
        if pathlength == phasecount:
            print('Now finally entered a loop where paths = phases')
            for phase, step in zip(phase_loop, curr_path):
                x = step[0]
                y = step[1]
                fieldnumber = x + y*3
                location_matrix[fieldnumber, ((phases*count_paths)+phase)] = 1 # currstep = phases
    return location_matrix, total_steps 
 