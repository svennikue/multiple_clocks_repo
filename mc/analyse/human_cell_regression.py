#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:05:55 2025

elastic net regression

@author: Svenja Küchenhoff

replicating and somewhat adjusting El-Gaby's Figure 5 regression



"""
import numpy as np
import mc
import matplotlib.pyplot as plt
import os
import pickle
import time
from operator import itemgetter
import scipy.stats as st

#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
#from sklearn.linear_model import PoissonRegressor

### SETTINGS
data_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives"
group_folder = "/Users/xpsy1114/Documents/projects/multiple_clocks/data/ephys_humans/derivatives/group"
file_name_all_subj_reg_prep = f"prep_data_for_regression"

subjects = [f"{i:02}" for i in range(1, 55) if i not in [6, 9, 27, 44]]
save_results = True
tt = time.time()
alpha=0.001 ##0.01 used in El-gaby paper
l1_ratio= 0.01




if not os.path.isdir(group_folder):
    os.mkdir(group_folder)

reg_dir = f"{group_folder}/regression"
reg_fig_dir = f"{group_folder}/regression/figures"
if not os.path.isdir(reg_dir):
    os.mkdir(reg_dir)
if not os.path.isdir(reg_fig_dir):
    os.mkdir(reg_fig_dir)

if os.path.isfile(os.path.join(group_folder,file_name_all_subj_reg_prep)):
    with open(os.path.join(group_folder,file_name_all_subj_reg_prep), 'rb') as f:
        data_and_regressors = pickle.load(f)
        print(f"opened stored dataset")
else:  
    print(f"loading and running preprocessing of human cells")
    data = mc.analyse.helpers_human_cells.load_cell_data(data_folder, subjects)
    # PART 1 
    # prepare all regressors for the data I want, in the same array format as the cells.
    # any regressors I want (e.g. current location, next location, state etc)
    # for all grids, all subjects: 
        # simulate 4x 9 location neurons
        # fires when currently at location, when next reward, next next, and prev. reward location
    data_and_regressors = mc.analyse.helpers_human_cells.prep_regressors_for_neurons(data)
    
    if save_results == True:
        # save the all_modelled_data dict such that I don't need to always run it again.
        with open(os.path.join(group_folder,file_name_all_subj_reg_prep), 'wb') as f:
            pickle.dump(data_and_regressors, f)
        print(f"saved the modelled data as {group_folder}/{file_name_all_subj_reg_prep}")
          

corr_dict = {}

# do this for only one subject to try with.
# for sub in data_and_regressors: 
#for sub in ['sub-01']: 
for sub in data_and_regressors:
    tt=time.time()
    corr_dict[sub] = {}
    single_sub_dict = data_and_regressors[sub]
    # PART 2
    # run a regression per neuron (ALL neurons and ALL subjects)
    # load if already exists
    if os.path.isfile(os.path.join(group_folder,f"all_regs_all_cells_all_models_{sub}")):
        with open(os.path.join(group_folder,f"all_regs_all_cells_all_models_{sub}"), 'rb') as f:
            result_dict = pickle.load(f)
            print(f"opened stored dataset for {sub}") 
            
    else:
        result_dict = {}
        for cell_idx, cell in enumerate(single_sub_dict['neurons'][0]):
            # create one entry in result_dict per cell 
            curr_cell = f"{single_sub_dict['cell_labels'][cell_idx]}_{cell_idx}_sub-01"
            result_dict[curr_cell] = {}
            
            # check for unique grid combos 
            unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(single_sub_dict['reward_configs'], axis=0,
                                                                    return_index=True,
                                                                    return_inverse=True,
                                                                    return_counts=True)
        
            for left_out_grid_idx in idx_unique_grid: 
                test_grid_idx = np.where(idx_same_grids == idx_same_grids[left_out_grid_idx])[0]
                training_grid_idx=np.setdiff1d(np.arange(len(single_sub_dict['reward_configs'])),test_grid_idx)
                
                # depending on the permutations, change the train and test dataset.
                for entry in single_sub_dict:
                    # only the regressors I created.
                    if entry.endswith('reg'):
                        result_dict[curr_cell][entry] = []
                        # then choose which tasks to take and go and create regressors_training_task
                        # and neurons.
                        simulated_neurons_training_tasks = itemgetter(*training_grid_idx)(single_sub_dict[entry])
                        all_neurons_training_tasks = itemgetter(*training_grid_idx)(single_sub_dict['neurons'])
                        curr_neuron_training_tasks = [all_neurons[cell_idx] for all_neurons in all_neurons_training_tasks]
                        
                        X = np.transpose(np.concatenate(simulated_neurons_training_tasks, axis = 1))
                        y = np.concatenate(curr_neuron_training_tasks)
                        alpha=0.01
                        reg = ElasticNet(alpha=alpha, l1_ratio = l1_ratio, positive=True).fit(X, y)            
                        coeffs_flat=reg.coef_
                        
                        # save per neuron, for all models, across perms.
                        # rewrite!!
                        result_dict[curr_cell][entry].append(coeffs_flat)
            
        if save_results == True:
            # save the all_modelled_data dict such that I don't need to always run it again.
            with open(os.path.join(group_folder,f"all_regs_all_cells_all_models_{sub}"), 'wb') as f:
                pickle.dump(result_dict, f)
            print(f"saved the modelled data as {group_folder}/all_regs_all_cells_all_models_{sub}")
          

    # PART 3
    # correlation.
    for entry in single_sub_dict:
        # only the regressors I created.
        if entry.endswith('reg'):
            corr_dict[sub][entry] = {}
        
    for cell_idx, cell in enumerate(single_sub_dict['neurons'][0]):
        # create one entry in result_dict per cell 
        curr_cell = f"{single_sub_dict['cell_labels'][cell_idx]}_{cell_idx}_sub-01"
        
        # check for unique grid combos 
        unique_grids, idx_unique_grid, idx_same_grids, counts = np.unique(single_sub_dict['reward_configs'], axis=0,
                                                                return_index=True,
                                                                return_inverse=True,
                                                                return_counts=True)
        for model in corr_dict[sub]:
            corr_dict[sub][model][curr_cell] = np.zeros(len(unique_grids))
            
        for task_idx, left_out_grid_idx in enumerate(idx_unique_grid): 
            test_grid_idx = np.where(idx_same_grids == idx_same_grids[left_out_grid_idx])[0]
            all_neurons_heldouttasks = itemgetter(*test_grid_idx)(single_sub_dict['neurons'])
            curr_neuron_heldouttasks = [all_neurons[cell_idx] for all_neurons in all_neurons_heldouttasks]
            curr_neuron_heldouttasks_flat = np.concatenate(curr_neuron_heldouttasks)
            
            for model in result_dict[curr_cell]:
                
                simulated_neurons_test_grids = itemgetter(*test_grid_idx)(single_sub_dict[model])
                simulated_neurons_test_grids_flat = np.transpose(np.concatenate(simulated_neurons_test_grids, axis = 1))
                predicted_activity_curr_neuron = np.sum((result_dict[curr_cell][model]*simulated_neurons_test_grids_flat), axis = 1)
                #check with mohamady why this step - in particular why include the actual neural activity??
                predicted_activity_curr_neuron_scaled = predicted_activity_curr_neuron*(
                    np.mean(curr_neuron_heldouttasks_flat)/np.mean(predicted_activity_curr_neuron))
                
                Predicted_Actual_correlation=st.pearsonr(curr_neuron_heldouttasks_flat,predicted_activity_curr_neuron)[0]
                corr_dict[sub][model][curr_cell][task_idx] = Predicted_Actual_correlation
                
                
                


# finally, plot the distribution for each model                                      
bins=50

# first collapse across subjects.
# Initialize empty lists to store values
button_box_list, musicbox_list, state_list = [], [], []

# Loop through the nested dictionary
for subject in corr_dict.values():  # Iterate over subjects
    for reg_type, neurons in subject.items():  # Iterate over regression types
        for neuron_data in neurons.values():  # Iterate over neurons
            if reg_type == "buttonbox_reg":
                button_box_list.append(neuron_data)
            elif reg_type == "musicbox_reg":
                musicbox_list.append(neuron_data)
            elif reg_type == "state_reg":
                state_list.append(neuron_data)

# Convert to NumPy arrays
results_of_corr = {}
results_of_corr['button_box'] = np.concatenate(button_box_list) if button_box_list else np.array([])
results_of_corr['musicbox'] = np.concatenate(musicbox_list) if musicbox_list else np.array([])
results_of_corr['state'] = np.concatenate(state_list) if state_list else np.array([])



for model in results_of_corr:
    # something like this:
    corrs_allneurons=results_of_corr[model]
    plt.figure()
    plt.title(f"correlation between {model} and all neurons")
    plt.hist(corrs_allneurons,bins=bins,color='grey')
    #plt.xlim(-1,1)
    plt.axvline(0,color='black',ls='dashed')
    plt.tick_params(axis='both',  labelsize=20)
    plt.tick_params(width=2, length=6)
    plt.savefig(reg_fig_dir+model+'GLM_analysis_all_neurons.svg',\
                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    print(len(corrs_allneurons))
    print(st.ttest_1samp(corrs_allneurons,0))



for model in results_of_corr:
    # something like this:
    corrs_allneurons=results_of_corr[model]

    # Compute the t-test for the correlations
    ttest_result = st.ttest_1samp(corrs_allneurons, 0)
    p_value = ttest_result.pvalue
    
    # Determine significance level based on p-value
    if p_value < 0.001:
        significance = '***'
    elif p_value < 0.01:
        significance = '**'
    elif p_value < 0.05:
        significance = '*'
    else:
        significance = 'n.s.'
    
    # Create a figure with a larger size for better aesthetics
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(corrs_allneurons, bins=bins, color='skyblue', edgecolor='black')
    
    # Add a vertical dashed line at 0
    plt.axvline(0, color='black', linestyle='dashed', linewidth=2)
    
    # Add title and axis labels with increased font sizes
    plt.title(f"Correlation between {model} and all neurons", fontsize=22)
    plt.xlabel("Correlation coefficient", fontsize=20)
    plt.ylabel("Frequency", fontsize=20)
    
    # Adjust tick parameters for better readability
    plt.tick_params(axis='both', labelsize=16, width=2, length=6)
    
    # Annotate the figure with the significance level
    plt.text(0.95, 0.95, f"Significance: {significance}\n(p = {p_value:.3e})",
             transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    plt.tight_layout()
    plt.show()
    
    # Print additional output if needed
    print("Number of neurons:", len(corrs_allneurons))
    print(ttest_result)
    




# next do the prediction of neural activity based on the coefficient values.
# multiply the regressors for the left out task with the beta weights of the trained set
# actually this will be a weighted sum, so add up all the fake regressors so that I have a single time series
# correlation is noisy, so maybe just do the correlation on 4 state bins
# then repeat this for every held-out task, and average the value 
# for each cell 




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