#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 14:45:04 2025

@author: xpsy1114
"""
import numpy as np
from matplotlib import pyplot as plt


# ----- single draw of your statistic -----
def loo_mean_corr_once(n=6, d=9, rng=None):
    """
    Draw n independent d-vectors ~ N(0,1).
    For each i, correlate x_i with the mean of the other n-1 vectors (over the d components).
    Return the average of the n correlations.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = rng.standard_normal((n, d))

    # Center each row (Pearson r across the d components)
    Xc = X - X.mean(axis=1, keepdims=True)

    rs = []
    for i in range(n):
        # mean of the others (still a d-vector)
        others = np.delete(Xc, i, axis=0)
        m = others.mean(axis=0)

        # center both vectors across components for Pearson r
        xi = Xc[i] - Xc[i].mean()
        mc = m - m.mean()

        denom = np.linalg.norm(xi) * np.linalg.norm(mc)
        r = 0.0 if denom == 0 else float(np.dot(xi, mc) / denom)
        rs.append(r)
    return float(np.mean(rs))

# ----- faster vectorized version for many sims -----
def loo_mean_corr_vectorized(n=6, d=9, rng=None):
    """
    Same statistic, but computed without Python loops over i.
    """
    if rng is None:
        rng = np.random.default_rng()
    X = rng.standard_normal((n, d))
    Xc = X - X.mean(axis=1, keepdims=True)        # row-center for Pearson

    # Sum of all rows, then leave-one-out means
    S = Xc.sum(axis=0, keepdims=True)              # shape (1,d)
    M = (S - Xc) / (n - 1)                         # each row i is mean of "others"
    # center each M[i] across components
    Mc = M - M.mean(axis=1, keepdims=True)

    # also center each Xi across components (already row-centered, but keep consistent)
    Xcc = Xc - Xc.mean(axis=1, keepdims=True)

    num = np.sum(Xcc * Mc, axis=1)
    den = np.linalg.norm(Xcc, axis=1) * np.linalg.norm(Mc, axis=1)
    r_i = np.where(den == 0, 0.0, num / den)
    return float(np.mean(r_i))

# ----- Monte Carlo driver -----
def simulate_loo(n_sims=50_000, n=8, d=4, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.empty(n_sims, dtype=float)
    for k in range(n_sims):
        vals[k] = loo_mean_corr_vectorized(n=n, d=d, rng=rng)
    return vals

# Example run & summary
if __name__ == "__main__":
    vals = simulate_loo(n_sims=50_000, n=8, d=4, seed=42)
    print("mean:", vals.mean())
    print("sd:", np.std(vals, ddof=1))
    print("quantiles (2.5, 5, 50, 95, 97.5):", np.quantile(vals, [0.025, 0.05, 0.5, 0.95, 0.975]))

    # quick histogram
    plt.hist(vals, bins=100, density=True, alpha=0.75)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Average LOO correlation")
    plt.ylabel("Density")
    plt.title("Null distribution: LOO-mean vs held-out correlation (n=6, d=4)")
    plt.show()
    
    
    

# def avg_pairwise_corr():
#     # draw 6 vectors of length 4
#     X = np.random.randn(6, 4)
#     # center each vector
#     Xc = X - X.mean(axis=1, keepdims=True)
#     norms = np.linalg.norm(Xc, axis=1)

#     rs = []
#     for i in range(6):
#         for j in range(i+1, 6):
#             r = np.dot(Xc[i], Xc[j]) / (norms[i] * norms[j])
#             rs.append(r)
#     return np.mean(rs)

# # simulate
# N = 100_000
# samples = np.array([avg_pairwise_corr() for _ in range(N)])

# print("Mean:", np.mean(samples))
# print("Std:", np.std(samples))
# print("2.5%, 97.5% quantiles:", np.quantile(samples, [0.025, 0.975]))

# # histogram
# plt.figure()
# plt.hist(samples, bins=100, density=True, alpha=0.7)
# plt.axvline(0, color='k', linestyle='--')
# plt.xlabel("Average correlation")
# plt.ylabel("Density")
# plt.title("Null distribution of average pairwise correlations")
# plt.show()


