#Funkcje pomocnicze
#Logika obliczania kosztu MDL oraz inicjalizacja sub-centroidów

import numpy as np

def init_subcentroids(X):
    if len(X) == 0:
        return [np.zeros_like(X), np.zeros_like(X)]
    elif len(X) == 1:
        return [X[0], X[0]]

    idx1 = np.random.randint(len(X))
    c1 = X[idx1]

    dists = np.sum((X - c1)**2, axis=1)
    if np.sum(dists) == 0:
        c2 = X[0]
    else:
        probs = dists / np.sum(dists)
        idx2 = np.random.choice(len(X), p=probs)
        c2 = X[idx2]
    return [c1, c2]

def mdl_cost(X, mu, C):
    d = X.shape[1]
    coords = np.sort(np.unique(X))
    diffs = np.diff(coords)
    min_diff = np.min(diffs[diffs > 0]) if len(diffs[diffs>0]) > 0 else 1e-5

    floatprecision = -np.log(min_diff) if min_diff < 1 else 1.0
    floatcost = (np.max(X) - np.min(X)) / floatprecision

    modelcost = len(C) * d * floatcost
    idxcost = len(X) * np.log(len(C)) if len(C) > 0 else 0

    c_val = 0
    for i, cluster_points in enumerate(C):
        if len(cluster_points) > 0:
            c_val += np.sum((cluster_points - mu[i])**2)

    residualcost = (len(X) * d * np.log(2*np.pi) + c_val) / 2
    return modelcost + residualcost + idxcost