import numpy as np
from kstar_means import kstar_means
from sklearn.datasets import make_blobs, fetch_openml
import umap


def run_test():
    centers = [[-10, -10], [10, 10], [10, -10], [-10, 10], [0,0]]
    X, y_true = make_blobs(n_samples=500, centers=centers, cluster_std=0.40, random_state=0)

    centroids, clusters = kstar_means(X, patience=15)

    print(f"Liczba wykrytych klastrów: {len(centroids)}")
    print(f"Centroidy: \n{np.array(centroids)}")


def run_mnist_test():
    print("Pobieranie zbioru MNIST...")

    #max: 70 000
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X_sample = X[:50000]

    print("Redukcja wymiarów algorytmem UMAP...")

    # UMAP from section 4.2 article
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.0)
    X_umap = reducer.fit_transform(X_sample)

    centroids, clusters = kstar_means(X_umap, patience=20)

    print(f"\nLiczba wykrytych klastrów: {len(centroids)}")
    #Expected result: 10-11 clasters for full MNIST (70 000)

if __name__ == "__main__":
    #run_test()
    run_mnist_test()