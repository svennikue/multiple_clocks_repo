#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculation of RDMs from datasets
@author: heiko, benjamin

16.01.2024:
    changes made by Svenja Küchenhoff 
    added: cross-validation correlation
edit 03rd of june 2024 - correction of 
    rdm = x[int(len(x)/2):,0:int(len(x)/2)]
    rdm_corr = (rdm + np.transpose(rdm))/2
edit 19th of june 2024-
    add 'weight_crosscorr'

"""
from __future__ import annotations
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
from rsatoolbox.rdm.rdms import RDMs
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.rdm.combine import from_partials
from rsatoolbox.data import average_dataset_by
from rsatoolbox.util.rdm_utils import _extract_triu_
if TYPE_CHECKING:
    from rsatoolbox.data.base import DatasetBase
    from numpy.typing import NDArray


def calc_rdm(dataset, method='euclidean', descriptor=None, noise=None,
             cv_descriptor=None, prior_lambda=1, prior_weight=0.1, weighting=None):
    """
    calculates an RDM from an input dataset

    This should usually be called with the method and the descriptor argument
    to specify the dissimilarity measure and which observations in the dataset
    belong to which condition.

    Args:
        dataset (rsatoolbox.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        method (String):
            a description of the dissimilarity measure (e.g. 'Euclidean')
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            used only for Mahalanobis and Crossnobis estimators
            defaults to an identity matrix, i.e. euclidean distance

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat, ds_i in enumerate(dataset):
            if noise is None:
                rdms.append(calc_rdm(
                    ds_i, method=method,
                    descriptor=descriptor,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise,
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise[i_dat],
                    cv_descriptor=cv_descriptor,
                    prior_lambda=prior_lambda, prior_weight=prior_weight))
        if descriptor is None:
            rdm = concat(rdms)
        else:
            rdm = from_partials(rdms, descriptor=descriptor)
    else:
        if method == 'euclidean':
            rdm = calc_rdm_euclidean(dataset, descriptor)
        elif method == 'correlation':
            rdm = calc_rdm_correlation(dataset, descriptor)
        elif method == 'mahalanobis':
            rdm = calc_rdm_mahalanobis(dataset, descriptor, noise)
        elif method == 'crossnobis':
            rdm = calc_rdm_crossnobis(dataset, descriptor, noise,
                                      cv_descriptor)
        elif method == 'poisson':
            rdm = calc_rdm_poisson(dataset, descriptor,
                                   prior_lambda=prior_lambda,
                                   prior_weight=prior_weight)
        elif method == 'poisson_cv':
            rdm = calc_rdm_poisson_cv(dataset, descriptor,
                                      cv_descriptor=cv_descriptor,
                                      prior_lambda=prior_lambda,
                                      prior_weight=prior_weight)
        # added by S.K. 16.01.2023
        elif method == 'crosscorr':
            rdm = calc_rdm_crosscorr(dataset, descriptor, cv_descriptor)
        elif method == 'weight_crosscorr':
            rdm = calc_rdm_weight_crosscorr(dataset, descriptor, cv_descriptor, weighting)
        # end addition
        else:
            raise NotImplementedError
        if descriptor is not None:
            rdm.sort_by(**{descriptor: 'alpha'})
    return rdm


def calc_rdm_movie(
        dataset, method='euclidean', descriptor=None, noise=None,
        cv_descriptor=None, prior_lambda=1, prior_weight=0.1,
        time_descriptor='time', bins=None):
    """
    calculates an RDM movie from an input TemporalDataset

    Args:
        dataset (rsatoolbox.data.dataset.TemporalDataset):
            The dataset the RDM is computed from
        method (String):
            a description of the dissimilarity measure (e.g. 'Euclidean')
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            used only for Mahalanobis and Crossnobis estimators
            defaults to an identity matrix, i.e. euclidean distance
        time_descriptor (String): descriptor key that points to the time
            dimension in dataset.time_descriptors. Defaults to 'time'.
        bins (array-like): list of bins, with bins[i] containing the vector
            of time-points for the i-th bin. Defaults to no binning.

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with RDM movie
    """

    if isinstance(dataset, Iterable):
        rdms = []
        for i_dat, ds_i in enumerate(dataset):
            if noise is None:
                rdms.append(calc_rdm_movie(
                    ds_i, method=method,
                    descriptor=descriptor))
            elif isinstance(noise, np.ndarray) and noise.ndim == 2:
                rdms.append(calc_rdm_movie(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise))
            elif isinstance(noise, Iterable):
                rdms.append(calc_rdm_movie(
                    ds_i, method=method,
                    descriptor=descriptor,
                    noise=noise[i_dat]))
        rdm = concat(rdms)
    else:
        if bins is not None:
            binned_data = dataset.bin_time(time_descriptor, bins)
            splited_data = binned_data.split_time(time_descriptor)
            time = binned_data.time_descriptors[time_descriptor]
        else:
            splited_data = dataset.split_time(time_descriptor)
            time = dataset.time_descriptors[time_descriptor]

        rdms = []
        for dat in splited_data:
            dat_single = dat.convert_to_dataset(time_descriptor)
            rdms.append(calc_rdm(dat_single, method=method,
                                 descriptor=descriptor, noise=noise,
                                 cv_descriptor=cv_descriptor,
                                 prior_lambda=prior_lambda,
                                 prior_weight=prior_weight))

        rdm = concat(rdms)
        rdm.rdm_descriptors[time_descriptor] = time
    return rdm


def calc_rdm_euclidean(dataset, descriptor=None):
    """
    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM
    """

    measurements, desc = _parse_input(dataset, descriptor)
    sum_sq_measurements = np.sum(measurements**2, axis=1, keepdims=True)
    rdm = sum_sq_measurements + sum_sq_measurements.T \
        - 2 * np.dot(measurements, measurements.T)
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    return _build_rdms(rdm, dataset, 'squared euclidean', descriptor, desc)


def calc_rdm_correlation(dataset, descriptor=None):
    """
    calculates an RDM from an input dataset using correlation distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    ma, desc = _parse_input(dataset, descriptor)
    ma = ma - ma.mean(axis=1, keepdims=True)
    ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]
    rdm = 1 - np.einsum('ik,jk', ma, ma)
    return _build_rdms(rdm, dataset, 'correlation', descriptor, desc)


def calc_rdm_mahalanobis(dataset, descriptor=None, noise=None):
    """
    calculates an RDM from an input dataset using mahalanobis distance
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (rsatoolbox.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            default: identity matrix, i.e. euclidean distance

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if noise is None:
        return calc_rdm_euclidean(dataset, descriptor)
    measurements, desc = _parse_input(dataset, descriptor)
    noise = _check_noise(noise, dataset.n_channel)
    kernel = measurements @ noise @ measurements.T
    rdm = np.expand_dims(np.diag(kernel), 0) + \
        np.expand_dims(np.diag(kernel), 1) - 2 * kernel
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    return _build_rdms(
        rdm,
        dataset,
        'squared mahalanobis',
        descriptor,
        desc,
        noise=noise
    )


def calc_rdm_crossnobis(dataset, descriptor, noise=None,
                        cv_descriptor=None):
    """
    calculates an RDM from an input dataset using Cross-nobis distance
    This performs leave one out crossvalidation over the cv_descriptor.

    As the minimum input provide a dataset and a descriptor-name to
    define the rows & columns of the RDM.
    You may pass a noise precision. If you don't an identity is assumed.
    Also a cv_descriptor can be passed to define the crossvalidation folds.
    It is recommended to do this, to assure correct calculations. If you do
    not, this function infers a split in order of the dataset, which is
    guaranteed to fail if there are any unbalances.

    This function also accepts a list of noise precision matricies.
    It is then assumed that this is the precision of the mean from
    the corresponding crossvalidation fold, i.e. if multiple measurements
    enter a fold, please compute the resulting noise precision in advance!

    To assert equal ordering in the folds the dataset is initially sorted
    according to the descriptor used to define the patterns.

    Args:
        dataset (rsatoolbox.data.dataset.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        noise (numpy.ndarray):
            dataset.n_channel x dataset.n_channel
            precision matrix used to calculate the RDM
            default: identity matrix, i.e. euclidean distance
        cv_descriptor (String):
            obs_descriptor which determines the cross-validation folds

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    noise = _check_noise(noise, dataset.n_channel)
    if noise is None:
        noise = np.eye(dataset.n_channel)
    if descriptor is None:
        raise ValueError('descriptor must be a string! Crossvalidation' +
                         'requires multiple measurements to be grouped')
    datasetCopy = deepcopy(dataset)
    if cv_descriptor is None:
        cv_desc = _gen_default_cv_descriptor(datasetCopy, descriptor)
        datasetCopy.obs_descriptors['cv_desc'] = cv_desc
        cv_descriptor = 'cv_desc'
    datasetCopy.sort_by(descriptor)
    cv_folds = np.unique(np.array(datasetCopy.obs_descriptors[cv_descriptor]))
    rdms = []
    if (noise is None) or (isinstance(noise, np.ndarray) and noise.ndim == 2):
        for i_fold, fold in enumerate(cv_folds):
            data_test = datasetCopy.subset_obs(cv_descriptor, fold)
            data_train = datasetCopy.subset_obs(
                cv_descriptor,
                np.setdiff1d(cv_folds, fold)
            )
            measurements_train, _, _ = \
                average_dataset_by(data_train, descriptor)
            measurements_test, _, _ = \
                average_dataset_by(data_test, descriptor)
            rdm = _calc_rdm_crossnobis_single(
                measurements_train, measurements_test, noise)
            rdms.append(rdm)
    else:  # a list of noises was provided
        measurements = []
        variances = []
        for i, i_fold in enumerate(cv_folds):
            data = datasetCopy.subset_obs(cv_descriptor, i_fold)
            measurements.append(average_dataset_by(data, descriptor)[0])
            variances.append(np.linalg.inv(noise[i]))
        for i_fold in range(len(cv_folds)):
            for j_fold in range(i_fold + 1, len(cv_folds)):
                if i_fold != j_fold:
                    rdm = _calc_rdm_crossnobis_single(
                        measurements[i_fold], measurements[j_fold],
                        np.linalg.inv(
                            (variances[i_fold] + variances[j_fold]) / 2)
                        )
                    rdms.append(rdm)
    rdms = np.array(rdms)
    rdm = np.einsum('ij->j', rdms) / rdms.shape[0]
    return _build_rdms(
        rdm,
        datasetCopy,
        'crossnobis',
        descriptor,
        noise=noise,
        cv=cv_descriptor
    )


def calc_rdm_poisson(dataset, descriptor=None, prior_lambda=1,
                     prior_weight=0.1):
    """
    calculates an RDM from an input dataset using the symmetrized
    KL-divergence assuming a poisson distribution.
    If multiple instances of the same condition are found in the dataset
    they are averaged.

    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    measurements, desc = _parse_input(dataset, descriptor)
    measurements = (measurements + prior_lambda * prior_weight) \
        / (1 + prior_weight)
    kernel = measurements @ np.log(measurements).T
    rdm = np.expand_dims(np.diag(kernel), 0) + \
        np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
    rdm = _extract_triu_(rdm) / measurements.shape[1]
    return _build_rdms(rdm, dataset, 'poisson', descriptor, desc)


def calc_rdm_poisson_cv(dataset, descriptor=None, prior_lambda=1,
                        prior_weight=0.1, cv_descriptor=None):
    """
    calculates an RDM from an input dataset using the crossvalidated
    symmetrized KL-divergence assuming a poisson distribution

    To assert equal ordering in the folds the dataset is initially sorted
    according to the descriptor used to define the patterns.

    Args:
        dataset (rsatoolbox.data.DatasetBase):
            The dataset the RDM is computed from
        descriptor (String):
            obs_descriptor used to define the rows/columns of the RDM
            defaults to one row/column per row in the dataset
        cv_descriptor (str): The descriptor that indicates the folds
            to use for crossvalidation

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDMs object with the one RDM

    """
    if descriptor is None:
        raise ValueError('descriptor must be a string! Crossvalidation' +
                         'requires multiple measurements to be grouped')
    dataset = deepcopy(dataset)
    if cv_descriptor is None:
        cv_desc = _gen_default_cv_descriptor(dataset, descriptor)
        dataset.obs_descriptors['cv_desc'] = cv_desc
        cv_descriptor = 'cv_desc'
    dataset.sort_by(descriptor)
    cv_folds = np.unique(np.array(dataset.obs_descriptors[cv_descriptor]))
    for i_fold in range(len(cv_folds)):
        fold = cv_folds[i_fold]
        data_test = dataset.subset_obs(cv_descriptor, fold)
        data_train = dataset.subset_obs(cv_descriptor,
                                        np.setdiff1d(cv_folds, fold))
        measurements_train, _, _ = average_dataset_by(data_train, descriptor)
        measurements_test, _, _ = average_dataset_by(data_test, descriptor)
        measurements_train = (measurements_train
                              + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        measurements_test = (measurements_test
                             + prior_lambda * prior_weight) \
            / (1 + prior_weight)
        kernel = measurements_train @ np.log(measurements_test).T
        rdm = np.expand_dims(np.diag(kernel), 0) + \
            np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
        rdm = _extract_triu_(rdm) / measurements_train.shape[1]
    return _build_rdms(rdm, dataset, 'poisson_cv', descriptor)


# addition by S.K. 16th of january 2024, edit 03rd of june 2024
def calc_rdm_crosscorr(dataset, descriptor=None, cv_descriptor=None):
    """
    calculates an RDM from an input dataset by creating a concatening the folds,
    creating a correlation matrix, and then averaging the lower triangle and 
    the top triangle of this nCond x nCond matrix by adding it to its transpose, 
    dividing by 2, and taking only the lower or upper triangle of the result.    
    
    """
    datasetCopy = deepcopy(dataset)
    if cv_descriptor is None:
        cv_desc = _gen_default_cv_descriptor(datasetCopy, descriptor)
        datasetCopy.obs_descriptors['cv_desc'] = cv_desc
        cv_descriptor = 'cv_desc'
    datasetCopy.sort_by(descriptor)
    cv_folds = np.unique(np.array(datasetCopy.obs_descriptors[cv_descriptor]))
    if len(cv_folds) > 2:
        print('Careful! there are more than 2 cv folds, but the function is written for only 2 folds.')
    elif len(cv_folds) == 2:
        data_test = datasetCopy.subset_obs(cv_descriptor, cv_folds[0])
        data_train = datasetCopy.subset_obs(
            cv_descriptor,
            np.setdiff1d(cv_folds, cv_folds[0])
        )
        ma_cv = np.concatenate((data_test.measurements, data_train.measurements), 0)
        ma_cv = ma_cv - ma_cv.mean(axis=1, keepdims=True)
        ma_cv /= np.sqrt(np.einsum('ij,ij->i', ma_cv, ma_cv))[:, None]       
        rdm_cv = 1 - np.einsum('ik,jk', ma_cv, ma_cv)  
        # what it should actually be
        rdm = rdm_cv[int(len(rdm_cv)/2):,0:int(len(rdm_cv)/2)]
        rdm = (rdm + np.transpose(rdm))/2
        
        # what it previously was
        # rdm = (rdm_cv[0:int(len(rdm)/2), int(len(rdm)/2):] + np.transpose(rdm_cv[0:int(len(rdm)/2), int(len(rdm)/2):]))/2
        # rdm = rdm[0:int(len(rdm)/2), int(len(rdm)/2):]
    
    return _build_rdms(rdm, dataset, 'crosscorr', descriptor)     


# second addition, for weighted neurons - 19th of June 2024
def calc_rdm_weight_crosscorr(dataset, descriptor=None, cv_descriptor=None, weighting=None):
    """
    computes similarity by doing: X_w_cov1 = X * diagWeights * transpose(X).
    
    Then averages the lower triangle of this nCond x nCond matrix by adding 
    it to its transpose, dividing by 2, and taking only the lower or 
    upper triangle of the result (i.e. the TH1 - TH2 correlations)
    
    """
    datasetCopy = deepcopy(dataset) 
    if cv_descriptor is None:
        cv_desc = _gen_default_cv_descriptor(datasetCopy, descriptor)
        datasetCopy.obs_descriptors['cv_desc'] = cv_desc
        cv_descriptor = 'cv_desc'
    datasetCopy.sort_by(descriptor)
    cv_folds = np.unique(np.array(datasetCopy.obs_descriptors[cv_descriptor]))
    if len(cv_folds) > 2:
        print('Careful! there are more than 2 cv folds, but the function is written for only 2 folds.')
    elif len(cv_folds) == 2:
        data_test = datasetCopy.subset_obs(cv_descriptor, cv_folds[0])
        data_train = datasetCopy.subset_obs(
            cv_descriptor,
            np.setdiff1d(cv_folds, cv_folds[0])
        )

        # concatenate both task halves
        ma_cv = np.concatenate((data_test.measurements, data_train.measurements), 0)
        # ma_cv has neurons as columns and rows as timepoints
        # deamean timpoints (substract mean of each row)
        ma_cv = ma_cv - ma_cv.mean(axis=1, keepdims=True) 
        # divide each timpoint by it's standard deviation
        ma_cv /= np.sqrt(np.einsum('ij,ij->i', ma_cv, ma_cv))[:, None]  


        # prep the weighting
        n_ring_neurons = 12
        # angles
        theta = np.linspace(0, 2 * np.pi, n_ring_neurons)  
        # make weights as long as my clocks model matrix
        theta_complete_clock = np.tile((theta), (1, 3*9))
        # theta_complete_clock = np.tile((theta), (3*9*2, 1)).flatten() 
        if weighting == 'sin':
            weights = np.sin(theta_complete_clock)
            diagWeights = np.diag(weights.flatten())
        if weighting == 'cos':
            weights = np.cos(theta_complete_clock)
            diagWeights = np.diag(weights.flatten())
        
        # compute RDM: 1 - covariance matrix
        # here I can include the weighting, instead of X XT
        rdm_cv = 1 - np.einsum('ik, kk, jk->ij', ma_cv, diagWeights, ma_cv)
        
        # only consider the lower square (TH 1 w TH 2) and make it symmetric
        rdm = rdm_cv[int(len(rdm_cv)/2):,0:int(len(rdm_cv)/2)]
        rdm = (rdm + np.transpose(rdm))/2


    return _build_rdms(rdm, dataset, 'crosscorr', descriptor)   


# end addition

def _calc_rdm_crossnobis_single(meas1, meas2, noise) -> NDArray:
    kernel = meas1 @ noise @ meas2.T
    rdm = np.expand_dims(np.diag(kernel), 0) + \
        np.expand_dims(np.diag(kernel), 1) - kernel - kernel.T
    return _extract_triu_(rdm) / meas1.shape[1]


def _gen_default_cv_descriptor(dataset, descriptor) -> np.ndarray:
    """ generates a default cv_descriptor for crossnobis
    This assumes that the first occurence each descriptor value forms the
    first group, the second occurence forms the second group, etc.
    """
    desc = dataset.obs_descriptors[descriptor]
    values, counts = np.unique(desc, return_counts=True)
    assert np.all(counts == counts[0]), (
        'cv_descriptor generation failed:\n'
        + 'different number of observations per pattern')
    n_repeats = counts[0]
    cv_descriptor = np.zeros_like(desc)
    for i_val in values:
        cv_descriptor[desc == i_val] = np.arange(n_repeats)
    return cv_descriptor


def _parse_input(
            dataset: DatasetBase,
            descriptor: Optional[str]
        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if descriptor is None:
        measurements = dataset.measurements
        desc = None
    else:
        measurements, desc, _ = average_dataset_by(dataset, descriptor)
    return measurements, desc


def _check_noise(noise, n_channel):
    """
    checks that a noise pattern is a matrix with correct dimension
    n_channel x n_channel

    Args:
        noise: noise input to be checked

    Returns:
        noise(np.ndarray): n_channel x n_channel noise precision matrix

    """
    if noise is None:
        pass
    elif isinstance(noise, np.ndarray) and noise.ndim == 2:
        assert np.all(noise.shape == (n_channel, n_channel))
    elif isinstance(noise, Iterable):
        for idx, noise_i in enumerate(noise):
            noise[idx] = _check_noise(noise_i, n_channel)
    elif isinstance(noise, dict):
        for key in noise.keys():
            noise[key] = _check_noise(noise[key], n_channel)
    else:
        raise ValueError('noise(s) must have shape n_channel x n_channel')
    return noise


def _build_rdms(
            utv: NDArray,
            ds: DatasetBase,
            method: str,
            obs_desc_name: str | None,
            obs_desc_vals: Optional[NDArray] = None,
            cv: Optional[NDArray] = None,
            noise: Optional[NDArray] = None
        ) -> RDMs:
    rdms = RDMs(
        dissimilarities=np.array([utv]),
        dissimilarity_measure=method,
        rdm_descriptors=deepcopy(ds.descriptors)
    )
    if (obs_desc_vals is None) and (obs_desc_name is not None):
        # obtain the unique values in the target obs descriptor
        _, obs_desc_vals, _ = average_dataset_by(ds, obs_desc_name)

    if _averaging_occurred(ds, obs_desc_name, obs_desc_vals):
        orig_obs_desc_vals = np.asarray(ds.obs_descriptors[obs_desc_name])
        for dname, dvals in ds.obs_descriptors.items():
            dvals = np.asarray(dvals)
            avg_dvals = np.full_like(obs_desc_vals, np.nan, dtype=dvals.dtype)
            for i, v in enumerate(obs_desc_vals):
                subset = dvals[orig_obs_desc_vals == v]
                if len(set(subset)) > 1:
                    break
                avg_dvals[i] = subset[0]
            else:
                rdms.pattern_descriptors[dname] = avg_dvals
    else:
        rdms.pattern_descriptors = deepcopy(ds.obs_descriptors)
    # Additional rdm_descriptors
    if noise is not None:
        rdms.descriptors['noise'] = noise
    if cv is not None:
        rdms.descriptors['cv_descriptor'] = cv
    return rdms


def _averaging_occurred(
            ds: DatasetBase,
            obs_desc_name: str | None,
            obs_desc_vals: NDArray | None
        ) -> bool:
    if obs_desc_name is None:
        return False
    orig_obs_desc_vals = ds.obs_descriptors[obs_desc_name]
    return len(obs_desc_vals) != len(orig_obs_desc_vals)
