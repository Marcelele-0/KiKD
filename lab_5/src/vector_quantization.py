import sys
from typing import Dict, Union

import numpy as np
from PIL import Image


def calculate_mse(original: np.ndarray, quantized: np.ndarray) -> float:
    """
    Calculates mean squared error (MSE) using float64 precision.
    """
    return float(np.mean((original.astype(np.float64) - quantized.astype(np.float64)) ** 2))


def calculate_snr(original: np.ndarray, mse: float) -> float:
    """
    Calculates signal-to-noise ratio (SNR) in dB.
    """
    if mse == 0:
        return float('inf')
    
    signal_power = np.mean(original.astype(np.float64) ** 2)
    return 10 * np.log10(signal_power / mse)


def get_nearest_centroids_manhattan(pixels: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    Finds nearest centroids optimized with chunking.
    """
    n_pixels = pixels.shape[0]
    labels = np.empty(n_pixels, dtype=np.int32)
    
    chunk_size = 32768 
    
    for i in range(0, n_pixels, chunk_size):
        end = min(i + chunk_size, n_pixels)
        batch = pixels[i:end] 
        dists = np.sum(np.abs(batch[:, np.newaxis, :] - codebook[np.newaxis, :, :]), axis=2)
        labels[i:end] = np.argmin(dists, axis=1)
        
    return labels


def lbg_algorithm(pixels: np.ndarray, target_k_exponent: int, epsilon: float = 0.01) -> np.ndarray:
    """
    LBG Algorithm optimized with Randomized Epsilon Splitting.
    """
    pixels = pixels.astype(np.float32)
    
    centroid_avg = np.mean(pixels, axis=0)
    codebook = np.array([centroid_avg], dtype=np.float32)
    
    current_k_exponent = 0
    
    # Generator liczb losowych (dla powtarzalności można dać seed, np. 42)
    rng = np.random.default_rng()
    
    while current_k_exponent < target_k_exponent:
        # --- Splitting Phase (ZMODYFIKOWANA) ---
        
        # Generujemy losowe wartości od 0.8*epsilon do 1.2*epsilon
        random_epsilon = rng.uniform(epsilon * 0.8, epsilon * 1.2, size=codebook.shape)
        
        cb_plus = codebook * (1.0 + random_epsilon)
        cb_minus = codebook * (1.0 - random_epsilon)
        
        codebook = np.vstack((cb_plus, cb_minus))
        current_k_exponent += 1
        
        # --- Optimization Phase (K-Means) ---
        max_iterations = 20
        min_change_threshold = 0.001
        
        for _ in range(max_iterations):
            labels = get_nearest_centroids_manhattan(pixels, codebook)
            
            n_centroids = len(codebook)
            counts = np.bincount(labels, minlength=n_centroids)
            
            safe_counts = counts.copy()
            safe_counts[safe_counts == 0] = 1
            
            new_codebook = np.empty_like(codebook)
            
            for ch in range(3):
                sum_channel = np.bincount(labels, weights=pixels[:, ch], minlength=n_centroids)
                new_codebook[:, ch] = sum_channel / safe_counts
            
            mask_empty = (counts == 0)
            if np.any(mask_empty):
                new_codebook[mask_empty] = codebook[mask_empty]
            
            shift = np.mean(np.abs(new_codebook - codebook))
            codebook = new_codebook
            
            if shift < min_change_threshold:
                break
            
    return codebook


def quantize_image(input_path: str, output_path: str, k_exponent: int) -> Dict[str, Union[float, int]]:
    """
    Main processing function.
    """
    print(f"Loading: {input_path}")
    img = Image.open(input_path).convert('RGB')
    width, height = img.size
    
    pixels_orig = np.array(img, dtype=np.float64)
    flat_pixels = pixels_orig.reshape(-1, 3)
    
    target_colors = 2 ** k_exponent
    unique_pixels_count = len(np.unique(flat_pixels, axis=0))
    
    if k_exponent >= 24 or target_colors >= unique_pixels_count:
        print(f"Target colors ({target_colors}) covers original count. Copying.")
        quantized_pixels = flat_pixels
    else:
        print(f"Running LBG for k={k_exponent} ({target_colors} colors)...")
        final_codebook = lbg_algorithm(flat_pixels, k_exponent)
        labels = get_nearest_centroids_manhattan(flat_pixels.astype(np.float32), final_codebook)
        quantized_pixels = final_codebook[labels]

    quantized_img_array = np.clip(quantized_pixels, 0, 255).reshape(height, width, 3).astype(np.uint8)
    
    out_img = Image.fromarray(quantized_img_array)
    out_img.save(output_path)
    print(f"Saved: {output_path}")
    
    mse = calculate_mse(flat_pixels, quantized_pixels)
    snr = calculate_snr(flat_pixels, mse)
    
    return {
        "mse": mse,
        "snr": snr,
        "colors_count": target_colors
    }


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python vector_quantization.py <input.tga> <output.tga> <K>")
        print("Where number of colors = 2^K")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        k_exponent = int(sys.argv[3])
        if not (0 <= k_exponent <= 24):
            raise ValueError("K must be between 0 and 24.")
    except ValueError as e:
        print(f"Argument error: {e}")
        sys.exit(1)
        
    try:
        res = quantize_image(input_file, output_file, k_exponent)
        
        print("\n--- Results ---")
        print(f"Number of colors: {res['colors_count']}")
        print(f"MSE: {res['mse']:.4f}")
        print(f"SNR: {res['snr']:.4f} dB")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()