import numpy as np


def train_lbg_1d(data: np.ndarray, num_bits: int, epsilon: float = 0.01) -> np.ndarray:
    """
    Trenuje kwantyzator nierównomierny (LBG) dla danych jednowymiarowych.
    Zwraca 'codebook' (tablicę wartości reprezentatywnych).
    """
    target_levels = 2 ** num_bits
    
    # Krok 0: Startujemy od średniej wszystkich danych
    codebook = np.array([np.mean(data)])

    # Pętla rozszczepiania (splitting)
    while len(codebook) < target_levels:
        # Rozszczepienie każdego centroidu na dwa (lekko w lewo i w prawo)
        lower = codebook * (1 - epsilon)
        upper = codebook * (1 + epsilon)
        
        # Łączymy i sortujemy, aby zachować porządek na osi liczbowej
        codebook = np.sort(np.concatenate([lower, upper]))

        # Optymalizacja (K-Means 1D)
        # W 1D przypisanie do najbliższego centroidu to po prostu sprawdzenie odległości.
        # Iterujemy kilka razy, aby centroidy "osiadły" w gęstych obszarach danych.
        for _ in range(10):
            # 1. Przypisz punkty do najbliższego centroidu
            # reshape(-1, 1) pozwala na broadcasting: (N, 1) - (1, K) -> (N, K)
            distances = np.abs(data.reshape(-1, 1) - codebook.reshape(1, -1))
            labels = np.argmin(distances, axis=1)

            # 2. Aktualizuj centroidy (średnia z przypisanych punktów)
            new_codebook = np.zeros_like(codebook)
            for i in range(len(codebook)):
                cluster = data[labels == i]
                if len(cluster) > 0:
                    new_codebook[i] = np.mean(cluster)
                else:
                    # Jeśli centroid jest pusty, zostawiamy go (lub można go przesunąć losowo)
                    new_codebook[i] = codebook[i]
            
            codebook = new_codebook
            
    return np.sort(codebook)


def quantize_1d(data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Zamienia wartości float na indeksy (uint8/16) najbliższego centroidu."""
    distances = np.abs(data.reshape(-1, 1) - codebook.reshape(1, -1))
    return np.argmin(distances, axis=1)


def dequantize_1d(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Odtwarza przybliżone wartości float na podstawie indeksów."""
    return codebook[indices]