#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:05:55 2025

elastic net regression

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