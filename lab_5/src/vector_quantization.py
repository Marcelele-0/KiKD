# src/vector_quantization.py
import sys
import numpy as np
from PIL import Image

def calculate_mse(original, quantized):
    """
    Oblicza błąd średniokwadratowy (MSE) między oryginałem a obrazem po kwantyzacji.
    """
    # Rzutujemy na float64, aby uniknąć przekręcenia licznika (overflow)
    # MSE = średnia z kwadratów różnic
    err = np.mean((original.astype(np.float64) - quantized.astype(np.float64)) ** 2)
    return err

def calculate_snr(original, mse):
    """
    Oblicza stosunek sygnału do szumu (SNR) w dB.
    """
    if mse == 0:
        return float('inf')
    
    # Moc sygnału = średnia kwadratów wartości pikseli oryginału
    signal_power = np.mean(original.astype(np.float64) ** 2)
    
    # SNR = 10 * log10(Moc_Sygnału / Moc_Błędu)
    return 10 * np.log10(signal_power / mse)

def get_nearest_centroids_manhattan(pixels, codebook):
    """
    Dla każdego piksela znajduje indeks najbliższego koloru z palety (codebook).
    Używa metryki taksówkowej: |dR| + |dG| + |dB|.
    """
    # pixels: (N, 3)
    # codebook: (K, 3)
    
    # Obliczamy różnicę każdego piksela z każdym centroidem (Broadcasting)
    # Wynik diff ma wymiar (N, K, 3)
    diff = np.abs(pixels[:, np.newaxis, :] - codebook[np.newaxis, :, :])
    
    # Sumujemy różnice po kanałach RGB (axis=2) -> Dystans Taksówkowy
    distances = np.sum(diff, axis=2)
    
    # Zwracamy indeksy centroidów o najmniejszym dystansie
    return np.argmin(distances, axis=1)

def lbg_algorithm(pixels, target_k_exponent, epsilon=0.01):
    """
    Algorytm Linde-Buzo-Gray (LBG).
    Argument 'target_k_exponent' to wykładnik (np. 4 oznacza 2^4 = 16 kolorów).
    """
    # 1. Inicjalizacja: Jeden centroid będący średnią wszystkich pikseli
    # axis=0 oznacza średnią po kolumnach (R, G, B)
    centroid_avg = np.mean(pixels, axis=0)
    codebook = np.array([centroid_avg])
    
    current_k_exponent = 0
    
    # Dopóki nie mamy wymaganej liczby bitów (potęgi dwójki)
    while current_k_exponent < target_k_exponent:
        # --- KROK 1: SPLITTING (Rozszczepianie) ---
        # Każdy obecny centroid C zamieniamy na dwa: C*(1+eps) i C*(1-eps)
        # Podwajamy w ten sposób rozmiar codebooka.
        cb_plus = codebook * (1 + epsilon)
        cb_minus = codebook * (1 - epsilon)
        codebook = np.vstack((cb_plus, cb_minus))
        
        current_k_exponent += 1
        
        # --- KROK 2: OPTYMALIZACJA (Iteracje K-Means) ---
        # Przesuwamy centroidy w stronę faktycznych skupisk pikseli.
        # W zadaniu nie podano warunku stopu, przyjmujemy stałą liczbę iteracji.
        max_iterations = 10
        
        for _ in range(max_iterations):
            # A. Przypisz piksele do najbliższych centroidów (metryka taksówkowa)
            labels = get_nearest_centroids_manhattan(pixels, codebook)
            
            # B. Oblicz nowe pozycje centroidów
            new_codebook = np.zeros_like(codebook)
            
            for i in range(len(codebook)):
                # Wybierz wszystkie piksele przypisane do klastra 'i'
                cluster_pixels = pixels[labels == i]
                
                if len(cluster_pixels) > 0:
                    # Nowy centroid to średnia arytmetyczna jego pikseli
                    new_codebook[i] = np.mean(cluster_pixels, axis=0)
                else:
                    # Jeśli centroid jest "martwy" (zero pikseli), zostawiamy go
                    # (lub można go losowo zresetować, tu zostawiamy dla prostoty)
                    new_codebook[i] = codebook[i]
            
            # Aktualizuj codebook
            codebook = new_codebook

    return codebook

def quantize_image(input_path, output_path, k_exponent):
    """
    Główna funkcja wykonująca zadanie.
    k_exponent: liczba bitów (np. 8), co daje 2^8 = 256 kolorów.
    """
    print(f"Wczytywanie: {input_path}")
    img = Image.open(input_path).convert('RGB')
    width, height = img.size
    
    # Konwersja na float64 dla dokładności obliczeń LBG
    # Spłaszczamy obraz do listy pikseli (N, 3)
    pixels_orig = np.array(img, dtype=np.float64)
    flat_pixels = pixels_orig.reshape(-1, 3)
    
    target_colors = 2 ** k_exponent
    
    # Zabezpieczenie: jeśli chcemy więcej kolorów niż jest pikseli, LBG nie ma sensu
    unique_pixels = np.unique(flat_pixels, axis=0)
    
    if k_exponent >= 24 or target_colors >= len(unique_pixels):
        print(f"Liczba kolorów ({target_colors}) pokrywa oryginał. Kopiowanie.")
        quantized_pixels = flat_pixels
    else:
        print(f"Uruchamianie LBG dla k={k_exponent} ({target_colors} kolorów)...")
        # 1. Znajdź najlepszą paletę (Codebook)
        final_codebook = lbg_algorithm(flat_pixels, k_exponent)
        
        # 2. Skwantyzuj obraz (przypisz każdemu pikselowi kolor z palety)
        labels = get_nearest_centroids_manhattan(flat_pixels, final_codebook)
        quantized_pixels = final_codebook[labels]
    
    # Rekonstrukcja obrazu (powrót do uint8)
    # Clip jest ważny, bo operacje float mogły dać np. 255.0001
    quantized_img_array = np.clip(quantized_pixels, 0, 255).reshape(height, width, 3).astype(np.uint8)
    
    # Zapis
    out_img = Image.fromarray(quantized_img_array)
    out_img.save(output_path)
    print(f"Zapisano: {output_path}")
    
    # Statystyki
    mse = calculate_mse(flat_pixels, quantized_pixels)
    snr = calculate_snr(flat_pixels, mse)
    
    return {
        "mse": mse,
        "snr": snr,
        "colors_count": target_colors
    }

def main():
    if len(sys.argv) != 4:
        print("Użycie: python vector_quantization.py <input.tga> <output.tga> <K>")
        print("Gdzie liczba kolorów = 2^K")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        k_exponent = int(sys.argv[3])
        if not (0 <= k_exponent <= 24):
            raise ValueError("K musi być między 0 a 24.")
    except ValueError as e:
        print(f"Błąd argumentu: {e}")
        sys.exit(1)
        
    try:
        res = quantize_image(input_file, output_file, k_exponent)
        
        print("\n--- Wyniki ---")
        print(f"Liczba kolorów: {res['colors_count']}")
        print(f"MSE: {res['mse']:.4f}")
        print(f"SNR: {res['snr']:.4f} dB")
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()