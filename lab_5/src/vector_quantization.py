# src/vector_quantization.py
import sys
from typing import Dict, Union

import numpy as np
from PIL import Image


def calculate_mse(original: np.ndarray, quantized: np.ndarray) -> float:
    """
    Calculates mean squared error (MSE) between original and quantized image.

    Args:
        original (np.ndarray): The original image pixels.
        quantized (np.ndarray): The quantized image pixels.

    Returns:
        float: The mean squared error.
    """
    err: float = np.mean((original.astype(np.float64) - quantized.astype(np.float64)) ** 2)
    return err


def calculate_snr(original: np.ndarray, mse: float) -> float:
    """
    Calculates signal-to-noise ratio (SNR) in dB.

    Args:
        original (np.ndarray): The original image pixels.
        mse (float): The mean squared error.

    Returns:
        float: The signal-to-noise ratio in dB.
    """
    if mse == 0:
        return float('inf')
    signal_power: float = np.mean(original.astype(np.float64) ** 2)
    return 10 * np.log10(signal_power / mse)


def get_nearest_centroids_manhattan(pixels: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    Finds index of nearest color from codebook for each pixel using Manhattan distance.

    Args:
        pixels (np.ndarray): The image pixels of shape (N, 3).
        codebook (np.ndarray): The codebook centroids of shape (K, 3).

    Returns:
        np.ndarray: Indices of the nearest centroids for each pixel.
    """
    diff: np.ndarray = np.abs(pixels[:, np.newaxis, :] - codebook[np.newaxis, :, :])
    distances: np.ndarray = np.sum(diff, axis=2)
    return np.argmin(distances, axis=1)


def lbg_algorithm(pixels: np.ndarray, target_k_exponent: int, epsilon: float = 0.01) -> np.ndarray:
    """
    Linde-Buzo-Gray (LBG) algorithm for vector quantization.

    Args:
        pixels (np.ndarray): The image pixels.
        target_k_exponent (int): The exponent for the number of colors (2^k).
        epsilon (float, optional): The splitting perturbation factor. Defaults to 0.01.

    Returns:
        np.ndarray: The final codebook.
    """
    centroid_avg: np.ndarray = np.mean(pixels, axis=0)
    codebook: np.ndarray = np.array([centroid_avg])
    current_k_exponent: int = 0
    while current_k_exponent < target_k_exponent:
        # Splitting step
        cb_plus: np.ndarray = codebook * (1 + epsilon)
        cb_minus: np.ndarray = codebook * (1 - epsilon)
        codebook = np.vstack((cb_plus, cb_minus))
        current_k_exponent += 1
        # Optimization (K-Means iterations)
        max_iterations: int = 10
        for _ in range(max_iterations):
            labels: np.ndarray = get_nearest_centroids_manhattan(pixels, codebook)
            new_codebook: np.ndarray = np.zeros_like(codebook)
            for i in range(len(codebook)):
                cluster_pixels: np.ndarray = pixels[labels == i]
                if len(cluster_pixels) > 0:
                    new_codebook[i] = np.mean(cluster_pixels, axis=0)
                else:
                    new_codebook[i] = codebook[i]
            codebook = new_codebook
    return codebook


def quantize_image(input_path: str, output_path: str, k_exponent: int) -> Dict[str, Union[float, int]]:
    """
    Main function for quantization.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the quantized image.
        k_exponent (int): Number of bits (e.g. 8 means 256 colors).

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing MSE, SNR, and color count.
    """
    print(f"Loading: {input_path}")
    img: Image.Image = Image.open(input_path).convert('RGB')
    width, height = img.size
    pixels_orig: np.ndarray = np.array(img, dtype=np.float64)
    flat_pixels: np.ndarray = pixels_orig.reshape(-1, 3)
    target_colors: int = 2 ** k_exponent
    unique_pixels: np.ndarray = np.unique(flat_pixels, axis=0)
    if k_exponent >= 24 or target_colors >= len(unique_pixels):
        print(f"Number of colors ({target_colors}) covers original. Copying.")
        quantized_pixels: np.ndarray = flat_pixels
    else:
        print(f"Running LBG for k={k_exponent} ({target_colors} colors)...")
        final_codebook: np.ndarray = lbg_algorithm(flat_pixels, k_exponent)
        labels: np.ndarray = get_nearest_centroids_manhattan(flat_pixels, final_codebook)
        quantized_pixels: np.ndarray = final_codebook[labels]
    quantized_img_array: np.ndarray = np.clip(quantized_pixels, 0, 255).reshape(height, width, 3).astype(np.uint8)
    out_img: Image.Image = Image.fromarray(quantized_img_array)
    out_img.save(output_path)
    print(f"Saved: {output_path}")
    mse: float = calculate_mse(flat_pixels, quantized_pixels)
    snr: float = calculate_snr(flat_pixels, mse)
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
    input_file: str = sys.argv[1]
    output_file: str = sys.argv[2]
    try:
        k_exponent: int = int(sys.argv[3])
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