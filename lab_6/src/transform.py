import numpy as np

def apply_haar_transform(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rozdziela obraz na pasmo dolne (średnie) i górne (szczegóły) parami pikseli.
    L[i] = (x[2i] + x[2i+1]) / 2
    H[i] = (x[2i] - x[2i+1]) / 2
    """
    # Spłaszczamy do 1D, żeby operować na parach pikseli w ciągu
    flat = channel.flatten().astype(float)
    
    # Jeśli liczba pikseli nieparzysta, odcinamy ostatni (dla uproszczenia labu)
    if len(flat) % 2 != 0:
        flat = flat[:-1]
        
    # Pobieramy co drugi piksel
    even = flat[0::2] # Parzyste
    odd = flat[1::2]  # Nieparzyste
    
    low = (even + odd) / 2.0
    high = (even - odd) / 2.0
    
    return low, high

def inverse_haar_transform(low: np.ndarray, high: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Rekonstruuje kanał z pasm L i H."""
    # Odwrócenie wzorów:
    # even = L + H
    # odd = L - H
    even = low + high
    odd = low - high
    
    # Przeplatanie (Interleaving)
    reconstructed = np.zeros(len(even) + len(odd))
    reconstructed[0::2] = even
    reconstructed[1::2] = odd
    
    # Przycinamy lub uzupełniamy do oryginalnego kształtu (jeśli coś ucięliśmy przy kodowaniu)
    target_size = original_shape[0] * original_shape[1]
    if len(reconstructed) < target_size:
        reconstructed = np.pad(reconstructed, (0, target_size - len(reconstructed)), 'edge')
    elif len(reconstructed) > target_size:
        reconstructed = reconstructed[:target_size]
        
    return np.clip(reconstructed.reshape(original_shape), 0, 255).astype(np.uint8)

def dpcm_encode(data: np.ndarray) -> np.ndarray:
    """
    Kodowanie różnicowe (DPCM).
    y[0] = x[0]
    y[i] = x[i] - x[i-1]
    """
    # Wstawiamy 0 na początek, żeby przesunąć tablicę
    preds = np.zeros_like(data)
    preds[1:] = data[:-1]
    return data - preds

def dpcm_decode(diffs: np.ndarray) -> np.ndarray:
    """Odkodowanie DPCM (Suma kumulacyjna)."""
    return np.cumsum(diffs)