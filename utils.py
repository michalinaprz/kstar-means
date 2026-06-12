#Funkcje pomocnicze
#Logika obliczania kosztu MDL oraz inicjalizacja sub-centroidów

import numpy as np

# funkcja do szukania dwóch kandydatów na nowe klastry
def init_subcentroids(X):
    if len(X) == 0:
        # jeżeli klaster jest pusty to zwraca dwa punkty złożone z samych zer
        return [np.zeros_like(X), np.zeros_like(X)]
    elif len(X) == 1:
        return [X[0], X[0]]

    #wybranie losowo pierwszego kandydata
    idx1 = np.random.randint(len(X))
    c1 = X[idx1]

    #obliczanie dystansu każdego punktu od wybranego juz kandydata
    dists = np.sum((X - c1)**2, axis=1) #Kwadrat Odległości Euklidesowej (odległość w linii prostej z twierdzenia Pitagorasa, ale bez wyciągania pierwiastka)
    if np.sum(dists) == 0:
        c2 = X[0]
    else:
        #tworzenie prawdopodobieństwa, im punkt jest dalej tym ma wieksze prawd do bycia wylosowanym na drugiego kandydata
        probs = dists / np.sum(dists)
        #zmuszamy punkt do odsuniecia sie od pierwszego jak najdalej
        idx2 = np.random.choice(len(X), p=probs)
        c2 = X[idx2]
    return [c1, c2]

#Całkowitą Długość Opisu (MDL), czyli ile "bitów informacji" potrzeba, żeby zapisać obecny układ
def mdl_cost(X, mu, C):
    d = X.shape[1] #obliczanie wszystkich wymiarów
    coords = np.sort(np.unique(X)) #posortowanie danych, usuniecie duplikatów
    diffs = np.diff(coords) #obliczamy różnicę miedzy sąsiadami
    min_diff = np.min(diffs[diffs > 0]) if len(diffs[diffs>0]) > 0 else 1e-5 #1e-5 ustawiamy jezeli wszystkie punkty na sobie leżą

    #potrzebujemy większej precyzji jeżeli bedzie kilka miejsc po przecinku
    floatprecision = -np.log(min_diff) if min_diff < 1 else 1.0
    # dzieli my floatprecii=sion przez rozpiętość danych (np.max(X) - np.min(X))
    floatcost = (np.max(X) - np.min(X)) / floatprecision

    #obliczanie ile razy trzeba ponieść ten koszt
    modelcost = len(C) * d * floatcost
    #idccost - koszt identyfikatorów, każdy z punktów musi wiedziec do ktorej grupy jest przypisany
    #np.log(len(C)) - ilosc bitów potrzebne do identyfikatora
    idxcost = len(X) * np.log(len(C)) if len(C) > 0 else 0

    #algorytm pamięta środek klastra i odchylenie punktów od srodka
    c_val = 0 #suma odległości wszystkich środków od ich centroida
    for i, cluster_points in enumerate(C):
        if len(cluster_points) > 0:
            c_val += np.sum((cluster_points - mu[i])**2)

    #wzór z artykułu - funkcja gęstości wyprowadzona z rozkładu normalnego
    #im bardziej punkty są rozstrzelone tym ta wartość jest większa
    residualcost = (len(X) * d * np.log(2*np.pi) + c_val) / 2
    return modelcost + residualcost + idxcost