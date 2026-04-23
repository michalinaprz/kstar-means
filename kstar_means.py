#Główna klasa

import numpy as np
from utils import init_subcentroids, mdl_cost
from logic import kmeans_step, maybe_split, maybe_merge


def kstar_means(X, patience=5):
    X = np.array(X)
    best_cost = np.inf
    unimproved_count = 0

    mu = [np.mean(X, axis=0)]
    C = [X]
    mu_s = [init_subcentroids(X)]

    dists1 = np.sum((X - mu_s[0][0]) ** 2, axis=1)
    dists2 = np.sum((X - mu_s[0][1]) ** 2, axis=1)
    labels_s = (dists2 < dists1).astype(int)
    c_s_initial = [[], []]
    for idx, pt in enumerate(X): c_s_initial[labels_s[idx]].append(pt)
    C_s = [[np.array(pts) if len(pts) > 0 else np.empty((0, X.shape[1])) for pts in c_s_initial]]

    while True:
        mu, C, mu_s, C_s = kmeans_step(X, mu, C, mu_s, C_s)
        mu, C, mu_s, C_s, did_split = maybe_split(X, mu, C, mu_s, C_s)
        if not did_split:
            mu, C, mu_s, C_s = kmeans_step(X, mu, C, mu_s, C_s)
            mu, C, mu_s, C_s = maybe_merge(X, mu, C, mu_s, C_s)

        cost = mdl_cost(X, mu, C)
        if cost < best_cost:
            best_cost = cost;
            unimproved_count = 0
        else:
            unimproved_count += 1
        if unimproved_count == patience: break

    return mu, C