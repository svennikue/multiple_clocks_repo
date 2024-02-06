#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:51:46 2024

@author: xpsy1114
"""

import numpy as np
import pandas as pd
import sklearn 

import matplotlib.pyplot as plt


# load iris dataset
from sklearn.utils import Bunch
from sklearn.datasets import load_iris

dataset = load_iris()
# A list of examples we want to select from the iris dataset for demonstration purposes.
sample = [ 57, 122, 118,  53, 117,  81,  70, 142,  84,  16, 103,  82,  66, 31,  83]

# np.take selects specific examples from the array representing the full dataset.
petal_length = np.take(dataset.data[:, 2], sample)
petal_width = np.take(dataset.data[:, 3], sample)
iris_data = Bunch(data=petal_length, target=petal_width)


import pandas as pd
df = pd.DataFrame(dataset.data, columns= dataset.feature_names)

# plot dataset
df.plot.scatter(x="sepal length (cm)", y="petal width (cm)")

i = 0
petal_length[i], petal_width[i]
# or print all at the same time
list(zip(petal_length, petal_width))
# plot again, but this for some reason looks different than before
plot_iris_dataset()

a = 0.1
b = 0
plot_iris_dataset_with_line(a, b)

plot_iris_dataset_with_line(a=0.1, b=0, residuals=True)

plot_iris_dataset_with_line(a=0.4, b=-0.2, residuals=True)

# write a MSE function
def mean_squared_error(a, b, data):
  x, y = data
  predicted = a * x + b
  truth = y
  squared_error = (predicted - truth)**2
  mse = np.mean(squared_error)
  return mse

data = (petal_length, petal_width)



# write a least squares estimate
def iris_least_squares_estimate(data):
  x, y = data 
  # TODO: compute the values for the expressions for a and b in the previous
  # section.
  a = None
  b = None
  
  return a, b


# Plotting the resulting line
data = (petal_length, petal_width)
a, b = iris_least_squares_estimate(data)
plot_iris_dataset_with_line(a=a, b=b)


# Computing the MSE of the resulting line
mean_squared_error(a, b, data)

# Get indices for five test points from the full dataset.
test_sample = [89, 20, 72, 67, 12]

test_petal_length = np.take(dataset.data[:, 2], test_sample)
test_petal_width = np.take(dataset.data[:, 3], test_sample)

plot_iris_dataset(test=True)

# TODO: call a function to plot the regression line for the test dataset.

def iris_least_squares_estimate(data):
    x, y = data 
    y_predicted = None # TODO: compute the prediction
    # solution:
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    a_nominator = (x -mean_x) * (y - mean_y)
    a_denominater = (x - mean_x)**2
    a = np.sum(a_nominator) / np.sum(a_denominater)
    b = mean_
    return a, b


predict_petal_widths(a, b) 


# Is the test MSE larger than the MSE you computed for the training data?
data = (test_petal_length, test_petal_width)
mean_squared_error() # TODO: pass the relevant arguments here.




#@title Helper functions

from sklearn.utils import Bunch
from sklearn.datasets import load_iris

dataset = load_iris()
sample = [ 57, 122, 118,  53, 117,  81,  70, 142,  84,  16, 103,  82,  66, 31,  83]
test_sample = [89, 20, 72, 67, 12]

petal_length = np.take(dataset.data[:, 2], sample)
petal_width = np.take(dataset.data[:, 3], sample)
iris_data = Bunch(data=petal_length, target=petal_width)

test_petal_length = np.take(dataset.data[:, 2], test_sample)
test_petal_width = np.take(dataset.data[:, 3], test_sample)

def plot_iris_dataset(a=None, b=None, model=None, residuals=False, test=False, savefig=False, figname=None):
  if test:
    plt.scatter(iris_data.data, iris_data.target, alpha=0.2, zorder=0)
    plt.scatter(test_petal_length, test_petal_width, c='red', zorder=0)
  else:
    plt.scatter(iris_data.data, iris_data.target, zorder=0)
  
  x = np.linspace(0, 8, 100).reshape(-1, 1)
  if model:
    plt.plot(x, model.predict(x), c='orange', zorder=0)
  elif a is not None and b is not None:
    plt.plot(x, a * x + b, c='orange', zorder=0)

  
  if residuals:
    if test:
      X = test_petal_length.reshape(-1, 1)
      y = test_petal_width
    else:
      X = iris_data.data.reshape(-1, 1)
      y = iris_data.target
    if model:
      y_pred = model.predict(X)
    elif a is not None and b is not None:
      y_pred = a * X + b
    plt.vlines(X, y, y_pred, colors='red', zorder=1)
  
  plt.xlabel("petal length (cm)")
  plt.ylabel("petal width (cm)")
  plt.xlim([-0.25, 8.25])
  plt.ylim([-0.25, 2.75])

  if savefig:
    plt.savefig('{}.jpeg'.format(figname if figname else 'figure'), dpi=600)

  plt.show()

def plot_iris_dataset_with_line(a, b, residuals=False, test=False, **kwargs):
  plot_iris_dataset(a=a, b=b, residuals=residuals, test=test, **kwargs)

# These functions are needed for the 3D dataset
def predict_house_price(x1, x2, a):
    y = a[0]*x1 + a[1]*x2 + a[2]
    return y

import plotly.graph_objects as go

def plot_3d_data_samples(data, a=None, plane=False):
  x, y, z = data

  fig = go.Figure()

  fig.add_trace(go.Scatter3d(
          x=x, y=y, z=z, mode='markers'))
  
  if plane:
    xmin = np.amin(x)
    xmax = np.amax(x)
    ymin = np.amin(y)
    ymax = np.amax(y)
    
    x1 = np.linspace(xmin, xmax, 100)
    x2 = np.linspace(ymin, ymax, 100)

    X1, X2 = np.meshgrid(x1, x2)
    Y = predict_house_price(X1, X2, a)

    fig.add_trace(go.Surface(
        x=x1, y=x2, z=Y))
    
  fig.update_layout(scene = dict(
      xaxis_title="Location",
      yaxis_title="Size (sqm)",
      zaxis_title="Price (k)"),
      width=700, 
      margin=dict(r=20, b=10, l=10, t=10)
  )
  fig.show()

def plot_3d_data_samples_with_fitted_plane(data, a):
  plot_3d_data_samples(data, a, plane=True)