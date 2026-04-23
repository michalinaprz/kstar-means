import numpy as np
from kstar_means import kstar_means
from sklearn.datasets import make_blobs


def run_test():
    #4 klastry
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

    print("Uruchamiam K*-means...")
    centroids, clusters = kstar_means(X)

    print(f"Liczba wykrytych klastrów: {len(centroids)}")
    print(f"Centroidy: \n{np.array(centroids)}")


if __name__ == "__main__":
    run_test()