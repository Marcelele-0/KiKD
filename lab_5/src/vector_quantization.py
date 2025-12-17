import sys
import numpy as np
from PIL import Image
import time

# Próba importu CuPy (dla GPU)
try:
    import cupy as cp
    print("Sukces: Wykryto CuPy! Obliczenia będą wykonywane na GPU.")
    USE_GPU = True
except ImportError:
    print("Uwaga: Nie wykryto biblioteki CuPy. Obliczenia na CPU (wolne!).")
    print("Zainstaluj: pip install cupy-cuda11x (lub 12x)")
    import numpy as cp  # Fallback: cp będzie udawało np
    USE_GPU = False

def calculate_mse(original, quantized):
    # MSE liczymy na końcu, może być na CPU (numpy), bo to jednorazowa akcja
    # Upewniamy się, że dane są na CPU (.get() jeśli to cupy array)
    if hasattr(original, 'get'): original = original.get()
    if hasattr(quantized, 'get'): quantized = quantized.get()
    
    err = np.mean((original.astype(np.float64) - quantized.astype(np.float64)) ** 2)
    return err

def calculate_snr(original, mse):
    if hasattr(original, 'get'): original = original.get()
    if mse == 0: return float('inf')
    signal_power = np.mean(original.astype(np.float64) ** 2)
    return 10 * np.log10(signal_power / mse)

def get_nearest_centroids_gpu_batched(pixels, codebook):
    """
    Wersja GPU z podziałem na partie (batches).
    Zapobiega błędowi 'Out Of Memory' przy dużym K.
    """
    n_pixels = pixels.shape[0]
    # Rozmiar partii - dostosuj do pamięci GPU (np. 4000-8000 jest bezpieczne dla 6-8GB VRAM)
    batch_size = 4096 
    
    labels = cp.empty(n_pixels, dtype=cp.int32)
    
    # Iterujemy kawałkami
    for i in range(0, n_pixels, batch_size):
        end = min(i + batch_size, n_pixels)
        batch = pixels[i:end]
        
        # 1. Broadcasting na małym wycinku (bezpieczne dla VRAM)
        # batch: (B, 1, 3), codebook: (1, K, 3) -> diff: (B, K, 3)
        diff = cp.abs(batch[:, cp.newaxis, :] - codebook[cp.newaxis, :, :])
        
        # 2. Suma (Taksówkowa)
        dists = cp.sum(diff, axis=2)
        
        # 3. Argmin
        labels[i:end] = cp.argmin(dists, axis=1)
        
        # (Opcjonalnie) Czyścimy pamięć podręczą co jakiś czas
        # cp.get_default_memory_pool().free_all_blocks() 
        
    return labels

def lbg_algorithm_gpu(pixels, target_k_exponent, epsilon=0.01):
    # 1. Inicjalizacja na GPU
    # Przenosimy dane na GPU tylko raz!
    pixels_gpu = cp.asarray(pixels)
    
    centroid_avg = cp.mean(pixels_gpu, axis=0)
    codebook_gpu = cp.array([centroid_avg])
    
    current_k_exponent = 0
    print(f"Start LBG (GPU): 1 -> {2**target_k_exponent} kolorów")
    
    while current_k_exponent < target_k_exponent:
        # --- SPLITTING ---
        cb_plus = codebook_gpu * (1 + epsilon)
        cb_minus = codebook_gpu * (1 - epsilon)
        codebook_gpu = cp.vstack((cb_plus, cb_minus))
        
        current_k_exponent += 1
        current_colors = len(codebook_gpu)
        
        sys.stdout.write(f"\r[Etap {current_k_exponent}/{target_k_exponent}] Rozszczepianie do {current_colors} kolorów... ")
        sys.stdout.flush()
        
        # --- OPTYMALIZACJA ---
        max_iterations = 10
        for it in range(max_iterations):
            # A. Przypisz (korzystając z funkcji batched)
            labels = get_nearest_centroids_gpu_batched(pixels_gpu, codebook_gpu)
            
            # B. Aktualizuj centroidy
            # Niestety pętla po centroidach w Pythonie jest wolna nawet z CuPy.
            # Ale przy 8192 kolorach GPU i tak nadrobi czas na liczeniu dystansów.
            
            new_codebook = cp.zeros_like(codebook_gpu)
            
            # Aby przyspieszyć, można by użyć kernelów CUDA, ale zostańmy przy czytelnym kodzie.
            # Ważne: Wszystkie operacje tutaj dzieją się wewnątrz VRAM.
            for i in range(len(codebook_gpu)):
                # Boolean indexing na GPU jest bardzo szybki
                mask = (labels == i)
                # Sprawdzamy czy są jakieś piksele (sumowanie booleanów daje liczbę true)
                if cp.any(mask):
                    new_codebook[i] = cp.mean(pixels_gpu[mask], axis=0)
                else:
                    new_codebook[i] = codebook_gpu[i]
            
            codebook_gpu = new_codebook
            
            sys.stdout.write(f"\r[Etap {current_k_exponent}/{target_k_exponent}] {current_colors} kol.: iter {it+1}/{max_iterations}")
            sys.stdout.flush()

    print("\nLBG zakończony.")
    
    # Na samym końcu zwracamy wynik (nadal na GPU, żeby szybko zrobić rekonstrukcję)
    return codebook_gpu

def quantize_image(input_path, output_path, k_exponent):
    print(f"Wczytywanie: {input_path}")
    img = Image.open(input_path).convert('RGB')
    width, height = img.size
    
    pixels_orig = np.array(img, dtype=np.float64)
    flat_pixels = pixels_orig.reshape(-1, 3)
    
    target_colors = 2 ** k_exponent
    unique_pixels = np.unique(flat_pixels, axis=0)
    
    if k_exponent >= 24 or target_colors >= len(unique_pixels):
        print("Liczba kolorów pokrywa oryginał. Kopiowanie.")
        quantized_pixels_np = flat_pixels
    else:
        # --- Start sekcji GPU ---
        start_time = time.time()
        
        # 1. LBG na GPU
        final_codebook_gpu = lbg_algorithm_gpu(flat_pixels, k_exponent)
        
        # 2. Rekonstrukcja na GPU (przypisanie po raz ostatni)
        # Przenosimy piksele na GPU (jeśli funkcja LBG tego nie zwróciła)
        pixels_gpu = cp.asarray(flat_pixels)
        final_labels = get_nearest_centroids_gpu_batched(pixels_gpu, final_codebook_gpu)
        
        # Fancy indexing na GPU
        quantized_pixels_gpu = final_codebook_gpu[final_labels]
        
        # 3. Powrót na CPU (dopiero teraz!)
        quantized_pixels_np = cp.asnumpy(quantized_pixels_gpu)
        
        end_time = time.time()
        print(f"Czas obliczeń na GPU: {end_time - start_time:.2f} s")
        # --- Koniec sekcji GPU ---

    quantized_img_array = np.clip(quantized_pixels_np, 0, 255).reshape(height, width, 3).astype(np.uint8)
    
    out_img = Image.fromarray(quantized_img_array)
    out_img.save(output_path)
    print(f"Zapisano: {output_path}")
    
    mse = calculate_mse(flat_pixels, quantized_pixels_np)
    snr = calculate_snr(flat_pixels, mse)
    
    return {"mse": mse, "snr": snr, "colors_count": target_colors}

def main():
    if len(sys.argv) != 4:
        print("Użycie: python vq_gpu.py <input.tga> <output.tga> <K>")
        sys.exit(1)
    
    # ... (reszta obsługi argumentów bez zmian) ...
    try:
        k = int(sys.argv[3])
        if not (0 <= k <= 24): raise ValueError
        res = quantize_image(sys.argv[1], sys.argv[2], k)
        print("\n--- Wyniki ---")
        print(f"MSE: {res['mse']:.4f}")
        print(f"SNR: {res['snr']:.4f} dB")
    except Exception as e:
        print(f"Błąd: {e}")

if __name__ == "__main__":
    main()