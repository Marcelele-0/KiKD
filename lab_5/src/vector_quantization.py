# src/vector_quantization.py
import sys
import numpy as np
from PIL import Image

def calculate_mse(original, quantized):
    """Błąd średniokwadratowy (MSE)."""
    # axis=None oznacza średnią ze wszystkich elementów macierzy
    return np.mean((original - quantized) ** 2)

def calculate_snr(original, mse):
    """Stosunek sygnału do szumu (SNR) w dB."""
    if mse == 0:
        return float('inf')
    
    # Moc sygnału = średnia kwadratów wartości oryginału
    signal_power = np.mean(original ** 2)
    
    # SNR = 10 * log10(Moc sygnału / Moc szumu)
    # Moc szumu to w przybliżeniu MSE
    return 10 * np.log10(signal_power / mse)

def get_nearest_centroids_manhattan(pixels, codebook):
    """
    Znajduje indeksy najbliższych centroidów dla każdego piksela 
    używając metryki taksówkowej (Manhattan).
    """
    # pixels: (N_pixels, 3)
    # codebook: (N_colors, 3)
    
    # Broadcasting do obliczenia różnic:
    # (N_pixels, 1, 3) - (1, N_colors, 3) -> (N_pixels, N_colors, 3)
    # To może być pamięciożerne dla dużych obrazów i dużych codebooków!
    # Dla celów labu przyjmujemy, że się zmieści w RAM.
    
    diff = np.abs(pixels[:, np.newaxis, :] - codebook[np.newaxis, :, :])
    distances = np.sum(diff, axis=2) # Suma po kanałach RGB
    
    # Zwraca indeksy najbliższych centroidów dla każdego piksela
    return np.argmin(distances, axis=1)

def lbg_algorithm(pixels, k_bits, epsilon=0.01):
    """Algorytm Linde-Buzo-Gray do kwantyzacji wektorowej."""
    
    # Krok 0: Inicjalizacja - jeden centroid (średnia z całego obrazu)
    codebook = np.mean(pixels, axis=0, keepdims=True)
    
    current_bits = 0
    
    while current_bits < k_bits:
        # Krok 1: Rozszczepianie (Splitting)
        # Każdy wektor y rozbijamy na y(1+eps) i y(1-eps)
        # Epsilon jest małym wektorem zaburzenia
        
        # Tworzymy dwa nowe wektory dla każdego starego
        cb_plus = codebook * (1 + epsilon)
        cb_minus = codebook * (1 - epsilon)
        
        # Łączymy je w nowy codebook (2x większy)
        codebook = np.vstack((cb_plus, cb_minus))
        current_bits += 1
        
        # Krok 2: Optymalizacja (iteracje K-Means z metryką taksówkową)
        # Wykonujemy kilka iteracji, aby dopasować centroidy
        max_iterations = 10 # Dla uproszczenia stała liczba iteracji
        prev_avg_dist = float('inf')
        
        for _ in range(max_iterations):
            # Przypisz piksele do najbliższych centroidów
            labels = get_nearest_centroids_manhattan(pixels, codebook)
            
            # Aktualizuj centroidy
            new_codebook = np.zeros_like(codebook)
            valid_centroids = []
            
            total_dist = 0
            
            for i in range(len(codebook)):
                # Wybierz piksele należące do klastra i
                cluster_pixels = pixels[labels == i]
                
                if len(cluster_pixels) > 0:
                    # Nowy centroid to średnia z pikseli w klastrze
                    new_codebook[i] = np.mean(cluster_pixels, axis=0)
                else:
                    # Jeśli do centroidu nic nie trafiło (martwy neuron), 
                    # zostawiamy go tam gdzie był (lub można losować)
                    new_codebook[i] = codebook[i]
            
            codebook = new_codebook
            
            # Prosty warunek stopu - jeśli zmiana jest mała (opcjonalne)
            # Tutaj pomijamy dla uproszczenia kodu
            
    return codebook

def quantize_image(input_path, output_path, bits):
    print(f"Wczytywanie: {input_path}")
    img = Image.open(input_path).convert('RGB')
    width, height = img.size
    
    # Spłaszcz obraz do tablicy pikseli (N, 3)
    pixels_orig = np.array(img, dtype=np.float64)
    flat_pixels = pixels_orig.reshape(-1, 3)
    
    # Jeśli bitów jest dużo (np. 24), po prostu kopiujemy obraz
    # LBG dla 16 milionów kolorów zabiłoby procesor w Pythonie
    if bits >= 24 or (2**bits) >= len(np.unique(flat_pixels, axis=0)):
        print("Liczba kolorów wystarczająca do pokrycia oryginału. Pomijam LBG.")
        quantized_pixels = flat_pixels
        final_codebook = None
    else:
        print(f"Uruchamianie LBG dla {bits} bitów ({2**bits} kolorów)...")
        final_codebook = lbg_algorithm(flat_pixels, bits)
        
        # Mapowanie końcowe
        labels = get_nearest_centroids_manhattan(flat_pixels, final_codebook)
        quantized_pixels = final_codebook[labels]
    
    # Rekonstrukcja obrazu
    quantized_img_array = quantized_pixels.reshape(height, width, 3).astype(np.uint8)
    
    # Zapis
    out_img = Image.fromarray(quantized_img_array)
    out_img.save(output_path)
    print(f"Zapisano: {output_path}")
    
    # Obliczenia błędów (na floatach dla precyzji)
    mse = calculate_mse(flat_pixels, quantized_pixels)
    snr = calculate_snr(flat_pixels, mse)
    
    return {
        "mse": mse,
        "snr": snr,
        "k_bits": bits,
        "num_colors": 2**bits if bits < 24 else "Original"
    }

def main():
    if len(sys.argv) != 4:
        print("Użycie: python vector_quantization.py <input.tga> <output.tga> <bity_koloru>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        bits = int(sys.argv[3])
        if not (0 <= bits <= 24):
            raise ValueError("Liczba bitów musi być między 0 a 24.")
    except ValueError as e:
        print(f"Błąd argumentu: {e}")
        sys.exit(1)
        
    try:
        res = quantize_image(input_file, output_file, bits)
        
        print("\n--- Wyniki Kwantyzacji ---")
        print(f"Błędy średniokwadratowy (MSE): {res['mse']:.4f}")
        print(f"Stosunek sygnału do szumu (SNR): {res['snr']:.4f} dB")
        
    except Exception as e:
        print(f"Wystąpił błąd: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()