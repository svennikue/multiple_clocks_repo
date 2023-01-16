#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:05:51 2023

@author: Svenja Kuechenhoff

this script calls on several simulation functions to eventually create RDMs
and check if predictions of location neuron activations are sufficiently 
distinct from phase clock neuron activation patterns.

"""

# %reset -f

import random
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from numpy import pi
from matplotlib.gridspec import GridSpec


## Section 1.
## Create the task
##
# first create the 3 x 3 grid and plot.
coord = [list(p) for p in product(range(3), range(3))]
cmap = cm.get_cmap('tab20b')
plt.scatter([x[0] for x in coord], [x[1] for x in coord], c=cmap(6), s=250)

# create 4 reward locations
points = random.sample(coord, 4)
# note that points[0:4] are my states: 
    # points[1] = A - dark red
    # points[2] = B - red
    # points[3] = C - medium red
    # points[4] = D - bright red
for i, x in enumerate(points):
    plt.scatter(x[0], x[1], c=cmap(i+11), s=250)

plt.yticks([0,1,2])
plt.xticks([0,1,2])
plt.grid(True)

