#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:05:55 2025

elastic net regression

<<<<<<< HEAD
@author: Svenja Küchenhoff

replicating and somewhat adjusting El-Gaby's Figure 5 regression



"""
import numpy as np
import mc
import matplotlib.pyplot as plt
import os
import pickle

#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
#from sklearn.linear_model import PoissonRegressor

### SETTINGS
data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
group_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
file_name_all_subj_reg_prep = f"prep_data_for_regression"

subjects = [f"{i:02}" for i in range(1, 55) if i not in [6, 9, 27, 44]]

save_results = True

if not os.path.isdir(group_folder):
    os.mkdir(group_folder)


if os.path.isfile(os.path.join(group_folder,file_name_all_subj_reg_prep)):
    with open(os.path.join(group_folder,file_name_all_subj_reg_prep), 'rb') as f:
        prep_data = pickle.load(f)
        print(f"opened stored dataset")
else:  
    print(f"loading and running preprocessing of human cells")
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)
    # PART 1 
    # prepare all regressors for the data I want, in the same array format as the cells.
    data_and_regressors = mc.analyse.helpers_human_cells.prep_regressors_for_neurons(data)
    
    if save_results == True:
        # save the all_modelled_data dict such that I don't need to always run it again.
        with open(os.path.join(group_folder,file_name_all_subj_reg_prep), 'wb') as f:
            pickle.dump(data_and_regressors, f)
        print(f"saved the modelled data as {group_folder}/{file_name_all_subj_reg_prep}")
  
        
  
    

# PART 1
# concatenating all repeats, all grids for
    # neurons
    # any regressors I want (e.g. current location, next location, state etc)
    # for all grids, all subjects: 
        # simulate 4x 9 location neurons
        # fires when currently at location, when next reward, next next, and prev. reward location
        
    
# PART 2
# run a regression per neuron (ALL neurons and ALL subjects! 
# careful, you have to simulate always from scratch per subj)
# exclude a task (a few tasks?) that you are using to predict, e.g. with np.setdiff1d
    #training_sessions=np.setdiff1d(np.arange(num_non_repeat_ses_found),ses_ind_ind_test)
    ##concatenating arrays
    #regressors_flat_trainingTasks_=regressors_flat_allTasks[training_sessions]
#


# alpha=0.01 ##0.01 used in paper
# X = regressors_flat
# y = Neuron_raw_eq_neuron_nonan
# reg = ElasticNet(alpha=alpha,positive=True).fit(X, y)            
# coeffs_flat=reg.coef_
# coeffs_all[neuron,ses_ind_ind_test]=coeffs_flat
# if Poisson_regression==True:
#     np.save(Input_folder+'Poisson_GLM_anchoring_coeffs_all_'+addition+mouse_recday+'.npy',coeffs_all)
# else:
#     np.save(Input_folder+'GLM_anchoring_coeffs_all_'+addition+mouse_recday+'.npy',coeffs_all)
    


# PART 3
# correlation.
# how exactly does this work??

# Qs for Mohamady: did you ever do only next goal location rather than next location?
# how did you do you 'phasic shift' of next locations?
# how many tasks concatenating and testing? what is best if it's not very precise?
# do you repeat this procedure with different left-out tasks or just one? yes! permute!!
# concatenate all of the same grid

# first find the goal-progress tuning
# average all neurons (binned) and find out for which bin it fires most
# but only if I include phase

# multiply the regressors for the left out task with the beta weights of the trained set
# actually this will be a weighted sum, so add up all the fake regressors so that I have a singel time series
# correlation is noisy, so maybe just do the correlation on 4 state bins
# then repeat this for every held-out task, and average the value 
# for each cell 

# consider doing this only on state cells
# scipy.stats.entropy(cell, 4)
# top 50% or top 75% to see if a cell cares for a certain state
# average across all tasks and bin the neurons


# check alignment first!!
# if state change and locations matches with timing of 
# state change times and grid rewards




# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import numpy as np
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # adjust alpha and l1_ratio based on your dataset
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# correlation = np.corrcoef(y_test, y_pred)[0, 1]
# print(f"Correlation: {correlation}")
=======
@author: xpsy1114
"""

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # adjust alpha and l1_ratio based on your dataset
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
correlation = np.corrcoef(y_test, y_pred)[0, 1]
print(f"Correlation: {correlation}")
>>>>>>> origin/main



# Elastic net regression is a regularization and variable selection method that combines the properties of both ridge and lasso regression. It is particularly useful when dealing with highly correlated data and when you want to balance between feature selection (like lasso) and regularization (like ridge).
# Key Components of Elastic Net Regression:

#     Ridge Regression: Adds a penalty equal to the square of the magnitude of coefficients; this type of regularization can shrink the coefficients but cannot set them to zero.
#     Lasso Regression: Adds a penalty equal to the absolute value of the magnitude of coefficients. This can shrink some coefficients completely to zero, which performs variable selection.

# Elastic net combines these two penalties. It has two parameters:

#     α (alpha): The mixing parameter between ridge (α = 0) and lasso (α = 1). It controls the balance of the two penalties.
#     λ (lambda): The regularization parameter that controls the overall strength of the penalty, hence the amount of shrinkage.

# How Elastic Net Works in Your Case:

# For neural timecourse data, using elastic net can help in determining which simulated neurons (regressors) contribute most significantly to predicting the activity of actual recorded neurons. It also helps in dealing with multicollinearity among the features (simulated neurons) and selecting the most relevant features by shrinking some coefficients to zero.
# Implementing Elastic Net in Python:

# Python's scikit-learn library provides an easy-to-use implementation of elastic net regression. Below is a step-by-step guide to applying elastic net to your data:

#     Prepare your data: Organize your data into a features matrix (X) where each column represents a simulated neuron's activity over time, and a target vector (y) representing the true neural activity you're trying to model.

#     Import necessary modules:

# from sklearn.linear_model import ElasticNet
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# import numpy as np

# Split the data into training and test datasets. Use the training data to fit your model and the test data to evaluate it.

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Elastic Net model:

# model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # adjust alpha and l1_ratio based on your dataset
# model.fit(X_train, y_train)

# Predict and evaluate:

#     y_pred = model.predict(X_test)
#     correlation = np.corrcoef(y_test, y_pred)[0, 1]
#     print(f"Correlation: {correlation}")

#     Iterate over all recorded neurons: Fit a model for each neuron individually as the target, and then analyze the distributions of the correlation values to examine the population shift or sparsity.

# Additional Tips:

#     Tuning Parameters: Use techniques like cross-validation to find the optimal values for alpha and l1_ratio. GridSearchCV or RandomizedSearchCV from scikit-learn can be helpful for this.
#     Model Evaluation: Besides correlation, consider using other metrics like RMSE (Root Mean Squared Error) or MAE (Mean Absolute Error) to assess model performance.
#     Feature Importance: After fitting the model, examine the coefficients (model.coef_) to determine which simulated neurons are most influential for predicting the activity of the recorded neurons.

# Implementing the above steps will allow you to effectively use elastic net regression to analyze neural timecourse data and draw meaningful conclusions about the influence of simulated neurons on the recorded neural responses.