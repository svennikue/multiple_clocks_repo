#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 10:57:00 2026

@author: xpsy1114
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def ols_beta_and_var(X, y):
    XtX = X.T @ X
    beta = inv(XtX) @ X.T @ y
    resid = y - X @ beta
    n, p = X.shape
    sigma2 = (resid @ resid) / (n - p)
    cov_beta = sigma2 * inv(XtX)
    return beta, cov_beta


def simulate_once(
    n_R=200, n_P=50,
    beta_true=1.0,
    noise_R=1.0,
    noise_P=3.0,
    seed=None
):
    rng = np.random.default_rng(seed)

    # Model of interest
    x_R = rng.normal(size=n_R)
    x_P = rng.normal(size=n_P)

    # One control regressor
    c_R = rng.normal(size=n_R)
    c_P = rng.normal(size=n_P)

    X_R = np.column_stack([x_R, c_R])
    X_P = np.column_stack([x_P, c_P])

    # Data
    y_R = beta_true * x_R + rng.normal(scale=noise_R, size=n_R)
    y_P = beta_true * x_P + rng.normal(scale=noise_P, size=n_P)

    # --- pooled regression (masked RDM style)
    X_pool = np.vstack([X_R, X_P])
    y_pool = np.concatenate([y_R, y_P])
    beta_pool, cov_pool = ols_beta_and_var(X_pool, y_pool)
    beta_pool = beta_pool[0]

    # --- split regressions
    beta_R, cov_R = ols_beta_and_var(X_R, y_R)
    beta_P, cov_P = ols_beta_and_var(X_P, y_P)

    bR, vR = beta_R[0], cov_R[0, 0]
    bP, vP = beta_P[0], cov_P[0, 0]

    # --- option 1: add betas
    beta_sum = bR + bP

    # --- option 2: inverse-variance weighted
    wR, wP = 1/vR, 1/vP
    beta_ivw = (wR*bR + wP*bP) / (wR + wP)

    return beta_pool, beta_sum, beta_ivw


def run_simulation(n_iter=5000):
    pooled = []
    summed = []
    ivw = []

    for i in range(n_iter):
        b_pool, b_sum, b_ivw = simulate_once(seed=i)
        pooled.append(b_pool)
        summed.append(b_sum)
        ivw.append(b_ivw)

    return np.array(pooled), np.array(summed), np.array(ivw)



pooled, summed, ivw = run_simulation()

# Equal noise & equal sizes → methods look more similar
simulate_once(n_R=100, n_P=100, noise_R=1, noise_P=1)

# Extreme imbalance → summed gets very noisy
simulate_once(n_R=300, n_P=20, noise_R=1, noise_P=5)




print("Means:")
print("Pooled:", pooled.mean())
print("Summed:", summed.mean())
print("IVW   :", ivw.mean())

print("\nStd dev:")
print("Pooled:", pooled.std())
print("Summed:", summed.std())
print("IVW   :", ivw.std())



plt.figure(figsize=(8, 5))
plt.hist(pooled, bins=50, alpha=0.5, label="Pooled")
plt.hist(ivw, bins=50, alpha=0.5, label="IVW")
plt.hist(summed, bins=50, alpha=0.5, label="Summed")
plt.axvline(1.0, color="k", linestyle="--", label="True beta")
plt.legend()
plt.xlabel("Estimated beta")
plt.ylabel("Count")
plt.title("Comparison of combination strategies")
plt.show()


