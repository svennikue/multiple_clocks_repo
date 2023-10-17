#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:39:55 2023

@author: xpsy1114
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import rsatoolbox
import rsatoolbox.data as rsd # abbreviation to deal with dataset
import os
from fsl.data.image import Image


channel_names = ['Oz', 'O1', 'O2', 'PO3', 'PO4', 'POz']  # channel names
stimulus = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4] # stimulus idx, each stimulus was presented twice

sampling_rate = 30 # in Hz
t = np.arange(-.5, 1.5, 1/sampling_rate) # time vector

n_observations = len(stimulus)
n_channels = len(channel_names)
n_timepoints = len(t)

measurements = np.random.randn(n_observations, n_channels, n_timepoints)  # random data

data = rsatoolbox.data.TemporalDataset(
    measurements,
    channel_descriptors={'names': channel_names},
    obs_descriptors={'stimulus': stimulus},
    time_descriptors={'time': t}
    )



data_dir = '/Users/xpsy1114/Documents/projects/multiple_clocks/data'
sub = '01'
file = os.path.join(data_dir, 'derivatives', f'sub-{sub}' ,'func', 'preproc_clean_01.feat','filtered_func_data_clean.nii.gz')

# loading the image using the fslpy toolbox
myimg = Image(file)





# HERE THE TUTORIAL STARTS

# this is basically already loading a searchlight with exactly 
# the right times/ voxels....
# HOW DOES THIS STEP WORK????
# actually, for me, this might be the moment when I combine both halfs of the data.
# this might potentially mean that I have to register them... but maybe not
# be careful with this!


path='/Users/xpsy1114/Documents/toolboxes/rsatoolbox-fmri/demos/'
searchlight_file = os.path.join(path, '92imageData/simTruePatterns.mat')
measurements = io.matlab.loadmat(searchlight_file)
measurements = measurements['simTruePatterns']
nCond = measurements.shape[0]
nVox = measurements.shape[1]

# plot the imported data
plt.imshow(measurements,cmap='gray')
plt.xlabel('Voxels')
plt.ylabel('Conditions')
plt.title('Measurements')

# this is supposed to be a 100 voxel (columns) x 92 conditions (rows) searchlight
# for me, this would be 100 voxels x timepoints (binned)

# now create a  dataset object
des = {'session': 1, 'subj': 1}
obs_des = {'conds': np.array(['cond_' + str(x) for x in np.arange(nCond)])}
chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
#obs_des = {'conds': np.array(['cond_' + str(x) for x in np.arange(1,nCond+1)])} # indices from 1
#chn_des = {'conds': np.array(['voxel' + str(x) for x in np.arange(1,nVox+1)])} # indices from 1
data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)

# second step: calculate dissimilarities of measures
# default is squared distance. non squared: rdm_nonsquare = rsatoolbox.rdm.sqrt_transform(rdm)
rdm = rsatoolbox.rdm.calc_rdm(data, method='euclidean', descriptor=None, noise=None)

# The Mahalanobis can be substantially more reliable than the standard Euclidean distance
# to use the Mahalonobis Distance, first estimate noise sigma
# doesnt say how it works though. I jumped over this, more on the website
# https://rsatoolbox.readthedocs.io/en/stable/distances.html
# noise_estim = rsd.noise.prec_from_measurements(data, obs_desc = obs_des)

# concatenate RDMs
# rsatoolbox.rdm.concat or rsatoolbox.rdm.append


# HOW TO CREATE SEARCHLIGHTS
# https://rsatoolbox.readthedocs.io/en/stable/demo_searchlight.html
import numpy as np
import matplotlib.pyplot as plt
from nilearn.image import new_img_like
import pandas as pd
import nibabel as nib
import seaborn as sns
from nilearn import plotting
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed
from rsatoolbox.rdm import RDMs
from glob import glob
from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight

tutorial_data_folder = os.path.join(path, 'subj02')
image_paths = list(glob(f"{tutorial_data_folder}/con_*.img"))
image_paths.sort()

def upper_tri(RDM):
    """upper_tri returns the upper triangular index of an RDM

    Args:
        RDM 2Darray: squareform RDM

    Returns:
        1D array: upper triangular vector of the RDM
    """
    # returns the upper triangle
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]

import matplotlib.colors
def RDMcolormapObject(direction=1):
    """
    Returns a matplotlib color map object for RSA and brain plotting
    """
    if direction == 0:
        cs = ['yellow', 'red', 'gray', 'turquoise', 'blue']
    elif direction == 1:
        cs = ['blue', 'turquoise', 'gray', 'red', 'yellow']
    else:
        raise ValueError('Direction needs to be 0 or 1')
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", cs)
    return cmap



# data set consist of one img-file per volume, so we need to loop over each file and add them to our data-array
# load one image to get the dimensions and make the mask
tmp_img = nib.load(image_paths[0])
# we infer the mask by looking at non-nan voxels
mask = ~np.isnan(tmp_img.get_fdata())
x, y, z = tmp_img.get_fdata().shape

# loop over all images
data = np.zeros((len(image_paths), x, y, z))
for x, im in enumerate(image_paths):
    data[x] = nib.load(im).get_fdata()

# only one pattern per image
image_value = np.arange(len(image_paths))



