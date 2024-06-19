#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 12:51:57 2024

This is script uses efficiency methods from Michel, Gramfort and Gervais from the 
nilearn toolbox, as well as on calc.py of rsatoolbox by Benjamin

It computes a data RDM per searchlight, based on a given similarity measure.
Input need to be the functional data projected to the surface (1 value per vertex)
and an adjacency matrix describing each searchlight.

@author: Svenja Küchenhoff
"""

from copy import deepcopy

import sys
import time
import warnings

import numpy as np
from joblib import Parallel, cpu_count, delayed
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

from nilearn.maskers.nifti_spheres_masker import _apply_mask_and_get_affinity

from .. import masking
from .._utils import check_niimg_4d, fill_doc
from ..image.resampling import coord_transform

ESTIMATOR_CATALOG = dict(svc=svm.LinearSVC, svr=svm.SVR)


@fill_doc



def search_light(
    X,
    A,
    method='crosscorr',
    cv=None,
    n_jobs=-1,
    verbose=0,
):
    """Compute RDM per search_light.

    Parameters
    ----------
    X : input dataset. comes in shape conditions x vertices (x task halves)

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.
        comes in shape vertices x vertices
        note:
            A.rows[list_i] is a list of vertices that will be computed at the same time.
        
    method (String): a description of the dissimilarity measure to compute 

    groups : array-like, optional, (default None)
        group label for each sample for cross validation.

        .. note::
            This will have no effect for scikit learn < 0.18

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

        
    %(n_jobs_all)s
    %(verbose0)s

    Returns
    -------
    data_RDM_all_searchlights : numpy.ndarray
        RDM array for each datapoint in X. dtype: float64.
    """
    
    group_iter = GroupIterator(A.shape[0], n_jobs)
    # this divides the searchlight computations into n_jobs that run in parallel.
    # A.shape[0] is the total number of vertices, group_iter defines which 
    # vertices will be run together.
    
    import pdb; pdb.set_trace() 
    
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        
        # choose simple correlation for simple distance measure
        if method == 'correlation':
            searchlight_data_RDM = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_group_iter_searchlightwise_corr)(
                    A.rows[list_i],
                    X,
                    thread_id + 1,
                    A.shape[0],
                    verbose,
                )
                for thread_id, list_i in enumerate(group_iter)
            )
        # choose crosscorr if distance between task halves is desired
        elif method == 'crosscorr':
            searchlight_data_RDM = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_group_iter_searchlightwise_crosscorr)(
                    A.rows[list_i],
                    X,
                    thread_id + 1,
                    A.shape[0],
                    verbose,
                )
                for thread_id, list_i in enumerate(group_iter)
            )
    # since there was parallel processing of many vertices, they need to 
    # be concatenated in the end.
    return np.concatenate(searchlight_data_RDM)


@fill_doc
class GroupIterator:
    """Group iterator.

    Provides group of features for search_light loop
    that may be used with Parallel.

    Parameters
    ----------
    n_features : int
        Total number of features
    %(n_jobs)s

    """

    def __init__(self, n_features, n_jobs=1):
        self.n_features = n_features
        if n_jobs == -1:
            n_jobs = cpu_count()
        self.n_jobs = n_jobs

    def __iter__(self):
        yield from np.array_split(np.arange(self.n_features), self.n_jobs)



def _group_iter_searchlightwise_crosscorr(
    list_rows,
    X,
    thread_id,
    total,
    verbose=0,
):
    """Perform grouped iterations of search_light RDM comp, method: cross-correlation.
    
    calculates an RDM from an input dataset by creating a concatening the folds,
    creating a correlation matrix, and then averaging the lower left square
    of this nCond x nCond matrix by adding it to its transpose, 
    dividing by 2, and taking only the lower triangle of the result.    
    
    Parameters
    ----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    X : input dataset. comes in shape conditions x vertices x task_halves

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    verbose : int, optional
        The verbosity level. Default is 0

    Returns
    -------
    data_RDM : numpy.ndarray
        RDM array for each voxel. dtype: float64.
    """
    
    data_RDMs_searchlights = np.zeros ((len(list_rows), int((len(X) * (len(X)-1) /2))))
    # lower triangle number 
    t0 = time.time()
    for i, row in enumerate(list_rows):
        datasetCopy = deepcopy(X[:, row, :])
        cv_folds = datasetCopy.shape[2]
        if cv_folds > 2:
            print('Careful! there are more than 2 cv folds, but the function is written for only 2 folds.')
        elif cv_folds == 2:
            # concatenate the dataset so you compute distance between folds
            data_test = datasetCopy[:,:,0]
            data_train = datasetCopy[:,:,1]
            ma_cv = np.concatenate((data_test, data_train), 0)
            
            # compute RDM pearson correlation stepwise
            ma_cv = ma_cv - ma_cv.mean(axis=1, keepdims=True)
            ma_cv /= np.sqrt(np.einsum('ij,ij->i', ma_cv, ma_cv))[:, None]       
            rdm_cv = 1 - np.einsum('ik,jk', ma_cv, ma_cv)  
            
            # average corr fold 1 vs fold 2
            rdm = rdm_cv[int(len(rdm_cv)/2):,0:int(len(rdm_cv)/2)]
            rdm = (rdm + np.transpose(rdm))/2
            
            # only store lower tril
            data_RDMs_searchlights[i] = rdm[np.triu_indices(len(rdm), 1)]

        if verbose > 0:
            # One can't print less than each 10 iterations
            step = 11 - min(verbose, 10)
            if i % step == 0:
                # If there is only one job, progress information is fixed
                crlf = "\r" if total == len(list_rows) else "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i}/{len(list_rows)} voxels "
                    f"({percent:0.2f}%, {remaining} seconds remaining){crlf}"
                )
    return data_RDMs_searchlights


def _group_iter_searchlightwise_corr(
    list_rows,
    X,
    thread_id,
    total,
    verbose=0,
):
    """Perform grouped iterations of search_light RDM comp, method: correlation.
    
    calculates an RDM from an input dataset using correlation distance   
    
    Parameters
    ----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    X : input dataset. comes in shape conditions x vertices x task_halves

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    verbose : int, optional
        The verbosity level. Default is 0

    Returns
    -------
    data_RDM : numpy.ndarray
        RDM array for each voxel. dtype: float64.
    """
    
    data_RDMs_searchlights = np.zeros ((len(list_rows), int((len(X) * (len(X)-1) /2))))
    # lower triangle number 
    t0 = time.time()
    for i, row in enumerate(list_rows):
        ma = deepcopy(X[:, row])
        ma = ma - ma.mean(axis=1, keepdims=True)
        ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]
        rdm = 1 - np.einsum('ik,jk', ma, ma)
        # only store lower tril
        data_RDMs_searchlights[i] = rdm[np.triu_indices(len(rdm), 1)]

        if verbose > 0:
            # One can't print less than each 10 iterations
            step = 11 - min(verbose, 10)
            if i % step == 0:
                # If there is only one job, progress information is fixed
                crlf = "\r" if total == len(list_rows) else "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i}/{len(list_rows)} voxels "
                    f"({percent:0.2f}%, {remaining} seconds remaining){crlf}"
                )
    return data_RDMs_searchlights




# CHECK IF I DO!!
# DONT KNOW IF I NEED THIS BIT?????


##############################################################################
# Class for search_light #####################################################
##############################################################################
@fill_doc
class SearchLight(BaseEstimator):
    """Implement search_light analysis using an arbitrary type of classifier.

    Parameters
    ----------
    mask_img : Niimg-like object
        See :ref:`extracting_data`.
        Boolean image giving location of voxels containing usable signals.

    process_mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Boolean image giving voxels on which searchlight should be
        computed.

    radius : float, optional
        radius of the searchlight ball, in millimeters. Defaults to 2.

    estimator : 'svr', 'svc', or an estimator object implementing 'fit'
        The object to use to fit the data
    %(n_jobs)s
    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(verbose0)s

    Notes
    -----
    The searchlight [Kriegeskorte 06] is a widely used approach for the
    study of the fine-grained patterns of information in fMRI analysis.
    Its principle is relatively simple: a small group of neighboring
    features is extracted from the data, and the prediction function is
    instantiated on these features only. The resulting prediction
    accuracy is thus associated with all the features within the group,
    or only with the feature on the center. This yields a map of local
    fine-grained information, that can be used for assessing hypothesis
    on the local spatial layout of the neural code under investigation.

    Nikolaus Kriegeskorte, Rainer Goebel & Peter Bandettini.
    Information-based functional brain mapping.
    Proceedings of the National Academy of Sciences
    of the United States of America,
    vol. 103, no. 10, pages 3863-3868, March 2006
    """

    def __init__(
        self,
        mask_img,
        process_mask_img=None,
        radius=2.0,
        estimator="svc",
        n_jobs=1,
        scoring=None,
        cv=None,
        verbose=0,
    ):
        self.mask_img = mask_img
        self.process_mask_img = process_mask_img
        self.radius = radius
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose

    def fit(self, imgs, y, groups=None):
        """Fit the searchlight.

        Parameters
        ----------
        imgs : Niimg-like object
            See :ref:`extracting_data`.
            4D image.

        y : 1D array-like
            Target variable to predict. Must have exactly as many elements as
            3D images in img.

        groups : array-like, optional
            group label for each sample for cross validation. Must have
            exactly as many elements as 3D images in img. default None
            NOTE: will have no effect for scikit learn < 0.18

        """
        # check if image is 4D
        imgs = check_niimg_4d(imgs)

        # Get the seeds
        process_mask_img = self.process_mask_img
        if self.process_mask_img is None:
            process_mask_img = self.mask_img

        # Compute world coordinates of the seeds
        process_mask, process_mask_affine = masking._load_mask_img(
            process_mask_img
        )
        process_mask_coords = np.where(process_mask != 0)
        process_mask_coords = coord_transform(
            process_mask_coords[0],
            process_mask_coords[1],
            process_mask_coords[2],
            process_mask_affine,
        )
        process_mask_coords = np.asarray(process_mask_coords).T

        X, A = _apply_mask_and_get_affinity(
            process_mask_coords,
            imgs,
            self.radius,
            True,
            mask_img=self.mask_img,
        )

        estimator = self.estimator
        if estimator == "svc":
            estimator = ESTIMATOR_CATALOG[estimator](dual=True)
        elif isinstance(estimator, str):
            estimator = ESTIMATOR_CATALOG[estimator]()

        scores = search_light(
            X,
            y,
            estimator,
            A,
            groups,
            self.scoring,
            self.cv,
            self.n_jobs,
            self.verbose,
        )
        scores_3D = np.zeros(process_mask.shape)
        scores_3D[process_mask] = scores
        self.scores_ = scores_3D
        return self


