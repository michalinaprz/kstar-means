#Główna klasa

import numpy as np
from utils import init_subcentroids, mdl_cost
from logic import kmeans_step, maybe_split, maybe_merge


def kstar_means(X, patience=5):
    X = np.array(X) #zamieniamy nasze dane na tablice numpy
    best_cost = np.inf #ustawiamy początkowy koszt na nieskończoność
    unimproved_count = 0 #ile razy z rzędu algorytm próbował coś zmienić

    mu = [np.mean(X, axis=0)] #początkowy środek ciężkości, czyli średnia ze wszystkich punktów
    C = [X] #C to lista wszystkich punktow przypisana do jednedo środka ciężkości (centroida)
    mu_s = [init_subcentroids(X)]
    #każdy główny klaster musi miec dwoch kandydatów na "zastępców" (sub-centroidy), losujemy je metodą k-means++ funkcją init_subcentroids

    #wyliczanie kwadratu odległości euklidesowej żeb przypisać punkty do dwóch wybranych sub-centroidów
    dists1 = np.sum((X - mu_s[0][0]) ** 2, axis=1)
    dists2 = np.sum((X - mu_s[0][1]) ** 2, axis=1)

    #tablica etykiet (zera i jedynki): Czy dystans do drugiego kandydata jest mniejszy niż do pierwszego? PRAWDA - 1, FAŁSZ - 0
    labels_s = (dists2 < dists1).astype(int)

    #segregujemy punkty na podstawie czy maja 0 czy 1
    c_s_initial = [[], []]
    for idx, pt in enumerate(X):
        c_s_initial[labels_s[idx]].append(pt)
    #zamieniamy tą segregacje na tablice numpy, zabezpieczenie przed zerową tablicą
    C_s = [[np.array(pts) if len(pts) > 0 else np.empty((0, X.shape[1])) for pts in c_s_initial]]

    best_mu, best_C = None, None  # Do tych zmiennych algorytm bedzie dodawał najlepszy wynik

    while True:
        # kmeans_step sprawdza dystans każdego punktu do obecnych środków ciężkości (mu) i na nowo przypisuje je do grup (C)
        # to samo robi dla sub-centroidów (mu_s i C_s)
        mu, C, mu_s, C_s = kmeans_step(X, mu, C, mu_s, C_s)

        # maybe_split sprawdza czy opłaca się rozbić klastry (czy koszt sie zmniejszy)
        # did_split - flaga prawda albo fałśz, zwraca info czy rzeczywiscie doszlo do podzialu
        mu, C, mu_s, C_s, did_split = maybe_split(X, mu, C, mu_s, C_s)

        #jeżeli nie podzieliło to algorytm upewnia się że środki są nadal na swoich miejscach
        #potem sprawdza czy opłaca się połączyć klastry (maybe_merge)
        if not did_split:
            mu, C, mu_s, C_s = kmeans_step(X, mu, C, mu_s, C_s)
            mu, C, mu_s, C_s = maybe_merge(X, mu, C, mu_s, C_s)

        #mdl - minimal description length
        # obliczanie kosztu, im niższy tym lepiej
        cost = mdl_cost(X, mu, C)

        #autorzy artykułu stwierdzili, że jeżeli koszt nie zmmiejsza się o wiecej niż 2 to model przestał się uczyć
        if best_cost - cost >= 2.0:
            # Jeśli koszt poprawił się o minimum 2 punkty w stosunku do poprzedniego rekordu to nadpisujemy go
            best_cost = cost
            # zerujemy licznik bo koszt się zmienił
            unimproved_count = 0
            # Kopiujemy najlepszy stan
            best_mu = [m.copy() for m in mu]
            best_C = [c.copy() for c in C]
        else:
            #jeżeli koszt nie poprawił się wystarczająco, dodajemy licznik
            unimproved_count += 1

        #jeżeli licznik braku zmiany kosztu jest równy cierpliwości algorytmu to stopujemy, czyli doszliśmy do najlepszego wyniku
        if unimproved_count >= patience:
            break

    # Zwracamy zapisany najlepszy stan, a nie ten z momentu przerwania pętli
    return best_mu if best_mu is not None else mu, best_C if best_C is not None else C