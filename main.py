import time
import numpy as np
import matplotlib.pyplot as plt
from kstar_means import kstar_means
from sklearn.datasets import make_blobs, fetch_openml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import umap


def get_predictions(X, centroids):
    """Przypisuje każdy punkt z X do najbliższego centroidu."""
    if len(centroids) == 0:
        return np.zeros(len(X))
    centroids = np.array(centroids)
    dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    #np.argmin(dists, axis=1): Wyciąga indeks (numer) tego centroidu, który był najbliżej
    return np.argmin(dists, axis=1)


def cluster_accuracy(y_true, y_pred):
    """Oblicza dokładność klastrowania (ACC) używając algorytmu węgierskiego."""
    #upewniamy się, że to liczby całkowite
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    #budowanie macierzy pomyłek/powiązań o rozmiarze DxD
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    #Ile razy algorytm wrzucił punkt do klastra i, podczas gdy w rzeczywistości był to punkt z klasy j?
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    #uruchomienie algorytmu węgierskiego — minimalizacja kosztów
    #ale my chcemy zmaksymalizować liczbę poprawnych rozwiązań, dlatego używamy w.max() - w: największe sukcesy stają się dla algorytmu "najniższym kosztem".
    #Algorytm analizuje macierz i znajduje najlepsze możliwe parowanie
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 100.0 / y_pred.size


def generate_centroids(k, d):
    """Funkcja do generowania danych syntetycznych, aby zachować minimalną odległość między środkami klastrów równą d."""
    centroids = []
    #losujemy pierwszy centroid w małym kwadracie od -d do d
    centroids.append(np.random.uniform(-d, d, size=2))

    max_attempts = 2000 #zabezpieczenie przed zapełnieniem planszy, jeżeli dystans jest zbyt duży, po 2000 razie nie bedzie probowal upchac klastrow jezeli nie bedzie miejsca
    attempts = 0 #ile razy wylosowano punkt ktory był za blisko innych klastrów

    while len(centroids) < k and attempts < max_attempts:
        ref_idx = np.random.randint(len(centroids)) #pobranie już istniejących klastrów jako punkty odniesienia
        ref_point = centroids[ref_idx]
        radius = np.random.uniform(d, 2 * d) #losujemy promień z przedziału od d do 2d
        angle = np.random.uniform(0, 2 * np.pi) #losowy kąt z 0 do 360
        #kandydta na nowy środek:
        new_point = ref_point + np.array([radius * np.cos(angle), radius * np.sin(angle)])

        #sprawdzenie, czy jest w dobrej odległości od innych klastrów
        distances = np.linalg.norm(np.array(centroids) - new_point, axis=1)
        if np.all(distances >= d):
            centroids.append(new_point)
            attempts = 0
        else:
            attempts += 1

    if len(centroids) < k:
        print(f"  [!] Ostrzeżenie: Udało się wygenerować tylko {len(centroids)} z {k} centroidów dla d={d}.")

    return np.array(centroids)


def run_synthetic_distance_test(results_table):
    print("\n--- Eksperyment: Wpływ odległości (d) na wykrywanie k (Tabela 2) ---")
    k_true = 20 #klastry
    distances = [2.0, 3.0, 4.0, 5.0] #sprawdzamy odległości
    n_samples_per_cluster = int(1000 / k_true) #tyle punktów na jeden klaster

    for d_min in distances:
        print(f"\n> Testowanie dla dystansu d = {d_min}...")
        centers = generate_centroids(k_true, d_min)

        #wygeneruje 50 punktow na klaster bo 1000/20 = 50
        X, y_true = make_blobs(n_samples=n_samples_per_cluster * k_true,
                               centers=centers,
                               cluster_std=1.0,
                               random_state=42)

        #obliczanie runtime algorytmu
        start_time = time.time()
        centroids, clusters = kstar_means(X, patience=15) #patience=15: jeśli przez 15 kroków koszt MDL nie spadnie, ma przestać szukać
        runtime = time.time() - start_time

        # Obliczanie metryk
        y_pred = get_predictions(X, centroids)
        acc = cluster_accuracy(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred) * 100
        nmi = normalized_mutual_info_score(y_true, y_pred) * 100

        # Dodanie do tabeli ze specjalnym formatowaniem, by zmieścić d
        results_table.append(f"Synth d={d_min}\t{acc:.2f}\t{ari:.2f}\t{nmi:.2f}\t{len(centroids)}\t{runtime:.2f}")

        # Rysowanie wykresu
        plot_results(clusters, centroids,
                     f"Syntetyczne (d={d_min}) - Wykryto {len(centroids)} klastrów (Prawdziwe k=20)")


def run_usps_test(results_table):
    print("\n--- Test na zbiorze USPS ---")
    X, y = fetch_openml('USPS', version=2, return_X_y=True, as_frame=False, parser='auto')

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.0)
    X_umap = reducer.fit_transform(X)

    start_time = time.time()
    centroids, clusters = kstar_means(X_umap, patience=20)
    runtime = time.time() - start_time

    # Obliczanie metryk
    y_pred = get_predictions(X_umap, centroids)
    acc = cluster_accuracy(y, y_pred)
    ari = adjusted_rand_score(y, y_pred) * 100
    nmi = normalized_mutual_info_score(y, y_pred) * 100

    results_table.append(f"USPS     \t{acc:.2f}\t{ari:.2f}\t{nmi:.2f}\t{len(centroids)}\t{runtime:.2f}")
    plot_results(clusters, centroids, f"USPS (UMAP 2D) - Wykryto {len(centroids)} klastrów")


def run_mnist_test(results_table):
    print("\n--- Test na zbiorze MNIST ---")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X_sample = X[:50000]
    y_sample = y[:50000]

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.0)
    X_umap = reducer.fit_transform(X_sample)

    start_time = time.time()
    centroids, clusters = kstar_means(X_umap, patience=20)
    runtime = time.time() - start_time

    # Obliczanie metryk
    y_pred = get_predictions(X_umap, centroids)
    acc = cluster_accuracy(y_sample, y_pred)
    ari = adjusted_rand_score(y_sample, y_pred) * 100
    nmi = normalized_mutual_info_score(y_sample, y_pred) * 100

    results_table.append(f"MNIST    \t{acc:.2f}\t{ari:.2f}\t{nmi:.2f}\t{len(centroids)}\t{runtime:.2f}")
    plot_results(clusters, centroids, f"MNIST (UMAP 2D) - Wykryto {len(centroids)} klastrów")


def plot_results(clusters, centroids, title):
    fig = plt.figure(figsize=(8, 6))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)

    for i, cluster in enumerate(clusters):
        if len(cluster) > 0:
            cluster = np.array(cluster)
            plt.scatter(cluster[:, 0], cluster[:, 1], s=15, alpha=0.8, label=f'Klaster {i}')

    centroids = np.array(centroids)
    if len(centroids) > 0:
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    c='black', marker='X', s=150, linewidths=2, zorder=10, label='Centroidy')

    plt.title(title, fontweight='bold')
    plt.xlabel("Wymiar 1")
    plt.ylabel("Wymiar 2")

    if len(centroids) <= 15:
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    results_table = []
    run_synthetic_distance_test(results_table)
    run_usps_test(results_table)
    run_mnist_test(results_table)

    print("\n" + "=" * 65)
    print("Dataset  \tACC\t\tARI\t\tNMI\t\tklastry\tRuntime (s)")
    print("-" * 65)
    for row in results_table:
        print(row)
    print("=" * 65)