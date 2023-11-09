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
import rsatoolbox.rdm as rsr


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


# create an example dataset with random data, subset some conditions
nChannel = 50
nObs = 12
randomData = np.random.rand(nObs, nChannel)
des = {'session': 1, 'subj': 1}
obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])}
chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nChannel)])}
data = rsd.Dataset(measurements=randomData,
                        descriptors=des,
                        obs_descriptors=obs_des,
                        channel_descriptors=chn_des
                        )
# select a subset of the dataset: select data only from conditions 0:4
sub_data = data.subset_obs(by='conds', value=[0,1,2,3,4])
data.obs_descriptors['conds']
sub_data.obs_descriptors['conds'] # without the 5

# Split by channels
nChannel = 3
nChannelVox = 10 # three ROIs, each with 10 voxels
nObs = 4
randomData = np.random.rand(nObs, nChannel*nChannelVox)
des = {'session': 1, 'subj': 1}
obs_des = {'conds': np.array([0, 1, 2, 3])}
chn_des = ['ROI1', 'ROI2', 'ROI3'] * nChannelVox
chn_des = {'ROIs': np.array(chn_des)}
data = rsd.Dataset(measurements=randomData,
                        descriptors=des,
                        obs_descriptors=obs_des,
                        channel_descriptors=chn_des
                        )
split_data = data.split_channel(by='ROIs')
data.channel_descriptors['ROIs']


# multi-subject dataset

# this is just a list of several datasets.

# create a datasets with random data
nVox = 50 # 50 voxels/electrodes/measurement channels
nCond = 10 # 10 conditions
nSubj = 5 # 5 different subjects
randomData = np.random.rand(nCond, nVox, nSubj)

obs_des = {'conds': np.array(['cond_' + str(x) for x in np.arange(nCond)])}
chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}

data = [] # list of dataset objects
for i in np.arange(nSubj):
    des = {'session': 1, 'subj': i+1}
    # append the dataset object to the data list
    data.append(rsd.Dataset(measurements=randomData[:,:,i],
                        descriptors=des,
                        obs_descriptors=obs_des,
                        channel_descriptors=chn_des
                        )
               )


# now estimate dissimilarities.
path='/Users/xpsy1114/Documents/toolboxes/rsatoolbox-fmri/demos/'
path_to_measurements = os.path.join(path, '92imageData/simTruePatterns.mat')
measurements = io.matlab.loadmat(path_to_measurements)
measurements = measurements['simTruePatterns'] #this is only 92,100
nCond = measurements.shape[0]
nVox = measurements.shape[1]
# now create a  dataset object
des = {'session': 1, 'subj': 1}
obs_des = {'conds': np.array(['cond_%02d' % x for x in np.arange(nCond)])}
chn_des = {'voxels': np.array(['voxel_' + str(x) for x in np.arange(nVox)])}
data = rsd.Dataset(measurements=measurements,
                   descriptors=des,
                   obs_descriptors=obs_des,
                   channel_descriptors=chn_des)

# calculate a RDM, where conditions is 'conds'
# conds is just the number of input rows
# default is euclidian distance.
RDM_euc = rsr.calc_rdm(data, descriptor='conds')
print(RDM_euc)
# alternatively can also take correlation distance
RDM_corr = rsr.calc_rdm(data, method='correlation', descriptor='conds')

# default is squared distance. non squared: rdm_nonsquare = rsatoolbox.rdm.sqrt_transform(rdm)
rdm = rsatoolbox.rdm.calc_rdm(data, method='euclidean', descriptor=None, noise=None)



# access content from RDM
dist_vectors = RDM_euc.get_vectors() # here a vector
dist_matrix = RDM_euc.get_matrices()
# visualise it 
fig, ax, ret_val = rsatoolbox.vis.show_rdm(RDM_euc)

# THIS IS ACTUALLY USEFUL!!
# to turn a differently computed RDM into an RDM object:
# create an RDM object with given entries:
dissimilarities = RDM_euc.get_vectors() # this has to be in the shape of 1 x e.g. 4186 (92x92/2 - diagonal (??))
RDM_euc_manual = rsr.RDMs(dissimilarities)


# crossvalidate dissimilarities
# first generate one dataset object with multiple measruments for each pattern behind each other
n_rep = 2
m_noisy = np.repeat(measurements, n_rep, axis=0)
m_noisy += np.random.randn(*m_noisy.shape)

conds = np.array(['cond_%02d' % x for x in np.arange(nCond)])
# session descriptor marks which measurement comes from which session
sessions = np.tile(np.arange(n_rep), 92)
conds = np.repeat(conds, n_rep)
obs_des = {'conds': conds, 'sessions': sessions}

des = {'subj': 1}

dataset = rsd.Dataset(
    measurements=m_noisy,
    descriptors=des,
    obs_descriptors=obs_des,
    channel_descriptors=chn_des)

# then, use crossnobis and define repeated measurements as 'session'
rdm_cv = rsatoolbox.rdm.calc_rdm(dataset, method='crossnobis', descriptor='conds', cv_descriptor='sessions')
rsatoolbox.vis.show_rdm(rdm_cv)

# diagonal covariance from measurements = univariate noise normalization

# the covariance is never used in rsatoolbox
# noise_cov_diag = rsatoolbox.data.noise.cov_from_measurements(dataset, obs_desc='conds', method='diag')
# computing the precision matrix (inverse of CoV) instead:
noise_prec_diag = rsatoolbox.data.noise.prec_from_measurements(dataset, obs_desc='conds', method='diag')

# shrinkage estimate from measurements = multivariate noise normalization
# 'shrinkage_eye' implements a shrinkage towards a multiple of the diagonal
noise_prec_shrink = rsatoolbox.data.noise.prec_from_measurements(dataset, obs_desc='conds', method='shrinkage_eye')
# 'shrinkage_diag' shrinks towards the data diagonal.
noise_prec_shrink = rsatoolbox.data.noise.prec_from_measurements(dataset, obs_desc='conds', method='shrinkage_diag')

# noise estimates based on residuals
# the residuals of a 1st level analysis to estimate the activations caused by the conditions
# put measurements you wish to use into nres x k a matrix, where k is the number of measurement channels

# example: (obviously do not use random residuals for this in applications)
residuals = np.random.randn(1000, dataset.n_channel) 
noise_pres_res = rsatoolbox.data.noise.prec_from_residuals(residuals)

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
tmp_img = nib.load(image_paths[0]) # one nifti of dims 53,63,46
# we infer the mask by looking at non-nan voxels
# bc only voxels of interest have values, all others are nan
mask = ~np.isnan(tmp_img.get_fdata()) # mask is boolean and has the same dims 53,63,46
x, y, z = tmp_img.get_fdata().shape

# loop over all images
data = np.zeros((len(image_paths), x, y, z))
for x, im in enumerate(image_paths):
    data[x] = nib.load(im).get_fdata()

# only one pattern per image
image_value = np.arange(len(image_paths))

# this is probably bc of spm: it gives one volume per x,y,z.
# so now data is 118 (timepoints) x 53 63 46 voxels.


# STEP 1: get searchlight centers and neighbors
# neighboring voxels = 5 voxels away from the central voxel
# at least 50% of the neighboring voxels needs to be within the brain mask (threshold=0.5)
centers, neighbors = get_volume_searchlight(mask, radius=5, threshold=0.5)

# STEP 2: get RDM for each voxel
# reshape data so we have n_observastions x n_voxels
data_2d = data.reshape([data.shape[0], -1])
data_2d = np.nan_to_num(data_2d) # now this is 118 timepoints x 153594 voxels
# Get RDMs
SL_RDM = get_searchlight_RDMs(data_2d, centers, neighbors, image_value, method='correlation')

# STEP 3: load model RDM and evaluate.
















































