#Kroki algorytmu
#Ewolucje klastrów (przypisanie, podział, łączenia)

import numpy as np
from numpy.ma.core import append

from utils import init_subcentroids

def kmeans_step(X, mu, C, mu_s, C_S):
    new_C = [[] for _ in mu]
    if len(mu) > 0:
        mu_arr = np.array(mu)
        dists = np.linalg.norm(X[:, np.newaxis] - mu_arr, axis=2)**2
        labels = np.argmin(dists, axis=1)
        for i, x in zip(labels, X):
            new_C[i].append(x)

    new_C = [np.array(pts) if len(pts) > 0 else np.empty((0, X.shape[1])) for pts in new_C]

    for i in range(len(mu)):
        if len(new_C[i]) > 0:
            mu[i] = np.mean(new_C[i], axis=0)

    new_C_s = []
    for i in range(len(mu)):
        sub_C = [[], []]

        if len(new_C[i]) > 0:
            sub_mu1, sub_mu2 = mu_s[i]
            dists1 = np.sum((new_C[i] - sub_mu1)**2, axis=1)
            dists2 = np.sum((new_C[i] - sub_mu2)**2, axis=1)
            labels_s = (dists2 < dists1).astype(int)
            for idx, pt in enumerate(new_C[i]):
                sub_C[labels_s[idx]].append(pt)

        sub_C = [np.array(pts) if len(pts) > 0 else np.empty((0, X.shape[1])) for pts in sub_C]
        new_C_s.append(sub_C)

        for j in range(2):
            if len(sub_C[j]) > 0:
                mu_s[i][j] = np.mean(sub_C[j], axis=0)

    return mu, new_C, mu_s, new_C_s


def maybe_split(X, mu, C, mu_s, C_s):
    best_costchange = 0
    split_at = -1

    for i in range(len(mu)):
        subc1, subc2 = C_s[i]
        submu1, submu2 = mu_s[i]
        sse_sub1 = np.sum((subc1 - submu1)**2) if len(subc1) > 0 else 0
        sse_sub2 = np.sum((subc2 - submu2)**2) if len(subc2) > 0 else 0
        sse_main = np.sum((C[i] - mu[i])**2) if len(C[i]) > 0 else 0

        costchange = sse_sub1 + sse_sub2 - sse_main + len(X) / (len(mu)+1)

        if costchange < best_costchange:
            best_costchange = costchange
            split_at = i

    if best_costchange < 0:
        new_mu1, new_mu2 = mu_s[split_at]
        mu.pop(split_at); mu.insert(split_at, new_mu1); mu.insert(split_at, new_mu2)
        C.pop(split_at); C.insert(split_at, C_s[split_at][1]); C.insert(split_at, C_s[split_at][0])
        mu_s.pop(split_at); C_s.pop(split_at);

        mu_s.insert(split_at, init_subcentroids(C[split_at]))
        mu_s.insert(split_at + 1, init_subcentroids(C[split_at + 1]))
        C_s.insert(split_at, [np.empty((0, X.shape[1])), np.empty((0, X.shape[1]))])
        C_s.insert(split_at + 1, [np.empty((0, X.shape[1])), np.empty((0, X.shape[1]))])

    return mu, C, mu_s, C_s, (best_costchange < 0)

def maybe_merge(X, mu, C, mu_s, C_s):
    if len(mu) < 2:
        return mu, C, mu_s, C_s
    min_dist = np.inf
    i1, i2 = -1, -1

    for i in range(len(mu)):
        for j in range(i+1, len(mu)):
            d = np.sum((mu[i] - mu[j])**2)
            if d < min_dist:
                min_dist = d
                i1, i2 = i, j

    Z = np.vstack((C[i1], C[i2])) if len(C[i1]) > 0 and len(C[i2]) > 0 else (C[i1] if len(C[i1]) > 0 else C[i2])
    m_merged = np.mean(Z, axis=0) if len(Z) > 0 else (mu[i1] + mu[i2]) / 2
    mainQ = np.sum((Z - m_merged) ** 2) if len(Z) > 0 else 0
    subcQ = (np.sum((C[i1] - mu[i1]) ** 2) if len(C[i1]) > 0 else 0) + (
        np.sum((C[i2] - mu[i2]) ** 2) if len(C[i2]) > 0 else 0)

    if mainQ - subcQ - len(X) / len(mu) < 0:
        new_sub_mu = [mu[i1], mu[i2]]
        new_sub_C = [C[i1], C[i2]]
        mu.pop(i2);
        C.pop(i2);
        mu_s.pop(i2);
        C_s.pop(i2)
        mu[i1] = m_merged;
        C[i1] = Z;
        mu_s[i1] = new_sub_mu;
        C_s[i1] = new_sub_C

    return mu, C, mu_s, C_s












