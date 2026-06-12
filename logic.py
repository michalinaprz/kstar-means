#Kroki algorytmu
#Ewolucje klastrów (przypisanie, podział, łączenia)

import numpy as np
from numpy.ma.core import append
from scipy.spatial.distance import cdist

from utils import init_subcentroids

def kmeans_step(X, mu, C, mu_s, C_S):
    # lista dla potencjalnych nowych klastrów
    new_C = [[] for _ in mu]
    if len(mu) > 0:
        #zamieniamy tablice ze srodkami ciężkości na tablice numpy
        mu_arr = np.array(mu)
        #obliczenie odległości pomiędzy wszystkimi punktami a wszystkimi środkami ciężkości
        dists = cdist(X, mu_arr, metric='sqeuclidean')
        #sprawdzamy która odległość była najmniejsza
        labels = np.argmin(dists, axis=1)
        #wrzucamy każdy punkt do odpowiedniego klastra
        for i, x in zip(labels, X):
            new_C[i].append(x)

    #zabezpiecza kod przed pustymi klastrami zamieniając je na wymiarowe tablice NumPy (żeby uniknąć błędów, jeśli klaster nagle stracił wszystkie punkty).
    new_C = [np.array(pts) if len(pts) > 0 else np.empty((0, X.shape[1])) for pts in new_C]

    #bierze wszystkie punkty przypisane do klastra i oblicza ich nowy środek ciężkości
    for i in range(len(mu)):
        if len(new_C[i]) > 0:
            mu[i] = np.mean(new_C[i], axis=0)

    #-----------DOKŁADNIE TO SAMO ALE DLA SUB-KLASTRÓW------------
    #robimy to po to aby algorytm był od razu gotowy do podziału
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

    #zwracanie aktualne listy
    return mu, new_C, mu_s, new_C_s


def maybe_split(X, mu, C, mu_s, C_s):
    # na początku zakładamy że zmiana kosztu to 0
    # interesują nas tylko koszty ujemne
    best_costchange = 0
    # indeks klastra który zostanie po podziale
    split_at = -1

    #pętla po klastrach
    for i in range(len(mu)):
        #aktualni zastępcy klastrów wybrani w pliku kstar_means
        subc1, subc2 = C_s[i]
        submu1, submu2 = mu_s[i]

        # sse - sum of squared errors, czyli jak bardzo punkty są "rosztrzelone" od środka
        # sse_sub1 i sse_sub2 - jak punkty byłyby rostrzelone gdyby rozdzielic klaster
        sse_sub1 = np.sum((subc1 - submu1)**2) if len(subc1) > 0 else 0
        sse_sub2 = np.sum((subc2 - submu2)**2) if len(subc2) > 0 else 0
        # sse_main - jak są rozstrzelone w obecnym klastrze
        sse_main = np.sum((C[i] - mu[i])**2) if len(C[i]) > 0 else 0

        # wzór z artykułu:
        # 0.5 * (sse_sub1 + sse_sub2 - sse_main) - zawsze będzie na minusie, czyli zysk
        # + len(X) / (len(mu) + 1) - kara za podział, koszt utworzenia nowego klastra
        costchange = 0.5 * (sse_sub1 + sse_sub2 - sse_main) + len(X) / (len(mu) + 1)
        # jeżeli costchange < 0 to podzial jest oplacalny

        # zapisanie tego klastra jako najlepszego
        if costchange < best_costchange:
            best_costchange = costchange
            split_at = i

    #jeżeli dzielimy klaster
    if best_costchange < 0:
        new_mu1, new_mu2 = mu_s[split_at]
        subc1, subc2 = C_s[split_at]

        mu.pop(split_at) #wyrzucamy starego klastra
        mu.insert(split_at, new_mu1) #wstawiamy nowych
        mu.insert(split_at + 1, new_mu2)

        C.pop(split_at) #wyrzucamy stara liste punktow (byly przypisane do starego klastra)
        C.insert(split_at, subc1) #tworzymy nowe listy
        C.insert(split_at + 1, subc2)

        #usuwamy stare klastry z ich miejsc i wykonujemy init_subcetroids do znalezienia ich nastepcow
        mu_s.pop(split_at)
        C_s.pop(split_at)

        mu_s.insert(split_at, init_subcentroids(C[split_at]))
        mu_s.insert(split_at + 1, init_subcentroids(C[split_at + 1]))
        C_s.insert(split_at, [np.empty((0, X.shape[1])), np.empty((0, X.shape[1]))])
        C_s.insert(split_at + 1, [np.empty((0, X.shape[1])), np.empty((0, X.shape[1]))])

    return mu, C, mu_s, C_s, (best_costchange < 0) #zwracanie nowych list i flagi czy doszlo do podziału (prawda/fałsz)

def maybe_merge(X, mu, C, mu_s, C_s):
    # zabezpieczenie, że nie wykona sie jezeli mamy mniej niz 2 klastry
    if len(mu) < 2:
        return mu, C, mu_s, C_s
    min_dist = np.inf #ustawiamy minimalny dystans na nieskonczonosc zeby nastepny ustawic na mniejszy
    i1, i2 = -1, -1 #kandydaci do połączenia

    #przechodzi przez wszystkie klastry i sprawdzanie jaka para jest najbliżej siebie
    for i in range(len(mu)):
        for j in range(i+1, len(mu)):
            d = np.sum((mu[i] - mu[j])**2)
            if d < min_dist:
                min_dist = d
                i1, i2 = i, j #najbliżsi sąsiedzi

    # Z - połączenie punktów w jedna wielką grupe
    Z = np.vstack((C[i1], C[i2])) if len(C[i1]) > 0 and len(C[i2]) > 0 else (C[i1] if len(C[i1]) > 0 else C[i2])
    # środek ciężkości połączonych punktów
    m_merged = np.mean(Z, axis=0) if len(Z) > 0 else (mu[i1] + mu[i2]) / 2

    #mainQ - suma błędów kwadratowych w nowym połączonym zbiorze
    mainQ = np.sum((Z - m_merged) ** 2) if len(Z) > 0 else 0
    #subQ - suma błędów kwadratowych jezeli zostałyby niepołączone
    subcQ = (np.sum((C[i1] - mu[i1]) ** 2) if len(C[i1]) > 0 else 0) + (
        np.sum((C[i2] - mu[i2]) ** 2) if len(C[i2]) > 0 else 0)

    # 0.5 * (mainQ - subcQ)$ - koszt za złe dopasowanie
    # - len(X) / len(mu) < 0 - zysk ze zwolnienia jednego klastra
    if 0.5 * (mainQ - subcQ) - len(X) / len(mu) < 0:
        # aktualizacja list
        # stare centroidy zamieniają się na zastępców
        new_sub_mu = [mu[i1], mu[i2]]
        new_sub_C = [C[i1], C[i2]]

        #kasowanie starego podziału
        mu.pop(i2); C.pop(i2); mu_s.pop(i2); C_s.pop(i2)

        #dodanie aktualnych podziałów
        mu[i1] = m_merged; C[i1] = Z;mu_s[i1] = new_sub_mu; C_s[i1] = new_sub_C

    return mu, C, mu_s, C_s












