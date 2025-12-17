import sys
from typing import Dict, Union

import numpy as np
from PIL import Image


def calculate_mse(original: np.ndarray, quantized: np.ndarray) -> float:
    """
    Calculates mean squared error (MSE) using float64 precision.
    """
    # float64 is required here for precision during squaring large arrays
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
    Finds nearest centroids.
    Optimized with chunking to fit into CPU L3 Cache and reduce RAM usage.
    """
    n_pixels = pixels.shape[0]
    n_codebook = codebook.shape[0]
    labels = np.empty(n_pixels, dtype=np.int32)
    
    chunk_size = 32768 
    
    for i in range(0, n_pixels, chunk_size):
        end = min(i + chunk_size, n_pixels)
        batch = pixels[i:end] # (Batch, 3)
        
        # Broadcasting logic:
        # (Batch, 1, 3) - (1, K, 3) -> (Batch, K, 3)
        # We assume codebook is small enough to keep broadcasting fast.
        
        # Manhattan Distance (|x1-x2| + |y1-y2|...)
        # Summing along axis 2 (RGB channels)
        dists = np.sum(np.abs(batch[:, np.newaxis, :] - codebook[np.newaxis, :, :]), axis=2)
        
        # Argmin finds the index of the minimum distance
        labels[i:end] = np.argmin(dists, axis=1)
        
    return labels


def lbg_algorithm(pixels: np.ndarray, target_k_exponent: int, epsilon: float = 0.01) -> np.ndarray:
    """
    LBG Algorithm fully optimized with NumPy vectorization.
    """
    # Ensure float32 for speed (SIMD friendly)
    pixels = pixels.astype(np.float32)
    
    # Initialize with global mean
    centroid_avg = np.mean(pixels, axis=0)
    codebook = np.array([centroid_avg], dtype=np.float32)
    
    current_k_exponent = 0
    
    while current_k_exponent < target_k_exponent:
        # --- Splitting Phase ---
        # Vectorized expansion of codebook
        # vstack is efficient enough here
        codebook = np.vstack((codebook * (1.0 + epsilon), codebook * (1.0 - epsilon)))
        current_k_exponent += 1
        
        # --- Optimization Phase (K-Means) ---
        max_iterations = 10
        
        for _ in range(max_iterations):
            # 1. Assignment (Expectation)
            labels = get_nearest_centroids_manhattan(pixels, codebook)
            
            # 2. Update (Maximization) - THE FASTEST NUMPY METHOD
            n_centroids = len(codebook)
            
            # np.bincount is extremely fast C-implementation for histograms.
            # We calculate "count" per cluster.
            counts = np.bincount(labels, minlength=n_centroids)
            
            # Avoid division by zero
            # We create a safe divisor (replace 0 with 1 to avoid warning, then mask later)
            safe_counts = counts.copy()
            safe_counts[safe_counts == 0] = 1
            
            # Sum pixels per cluster.
            # np.add.at is slow. 
            # Weighted np.bincount on each channel is much faster.
            new_codebook = np.empty_like(codebook)
            
            # Loop unrolling for RGB channels (only 3 iterations)
            for ch in range(3):
                # Calculate sum of pixel values for this channel, grouped by label
                sum_channel = np.bincount(labels, weights=pixels[:, ch], minlength=n_centroids)
                new_codebook[:, ch] = sum_channel / safe_counts
            
            # Handle empty clusters (where counts == 0)
            # Strategy: Keep the old centroid position if no pixels were assigned
            mask_empty = (counts == 0)
            if np.any(mask_empty):
                new_codebook[mask_empty] = codebook[mask_empty]
            
            codebook = new_codebook
            
    return codebook


def quantize_image(input_path: str, output_path: str, k_exponent: int) -> Dict[str, Union[float, int]]:
    """
    Main processing function.
    """
    print(f"Loading: {input_path}")
    img = Image.open(input_path).convert('RGB')
    width, height = img.size
    
    # Initial load. LBG will cast to float32 internally for speed.
    pixels_orig = np.array(img, dtype=np.float64)
    flat_pixels = pixels_orig.reshape(-1, 3)
    
    target_colors = 2 ** k_exponent
    
    # Check if quantization is needed
    # Using float32 view for uniqueness check is risky due to precision, so we use the double original
    unique_pixels_count = len(np.unique(flat_pixels, axis=0))
    
    if k_exponent >= 24 or target_colors >= unique_pixels_count:
        print(f"Target colors ({target_colors}) covers original count. Copying.")
        quantized_pixels = flat_pixels
        # No LBG needed
    else:
        print(f"Running optimized LBG for k={k_exponent} ({target_colors} colors)...")
        
        # 1. Find Codebook
        final_codebook = lbg_algorithm(flat_pixels, k_exponent)
        
        # 2. Quantize (Map pixels to codebook)
        # We need one final nearest neighbor search
        labels = get_nearest_centroids_manhattan(flat_pixels.astype(np.float32), final_codebook)
        quantized_pixels = final_codebook[labels]

    # Reconstruct
    quantized_img_array = np.clip(quantized_pixels, 0, 255).reshape(height, width, 3).astype(np.uint8)
    
    out_img = Image.fromarray(quantized_img_array)
    out_img.save(output_path)
    print(f"Saved: {output_path}")
    
    # Calculate stats on high precision data
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