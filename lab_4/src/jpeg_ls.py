import sys
import numpy as np
from PIL import Image
import math


def calculate_entropy(data: np.ndarray) -> float:
    """
    Calculate the Shannon entropy for an array of data (values 0-255).
    Args:
        data (np.ndarray): Input array.
    Returns:
        float: Calculated entropy.
    """
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def get_predictors(W: np.ndarray, N: np.ndarray, NW: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return a dictionary with results of 8 predictors for JPEG-LS.
    Args:
        W, N, NW (np.ndarray): Neighboring pixel arrays.
    Returns:
        dict[str, np.ndarray]: Predictor names and their arrays.
    """
    preds = {}
    preds['1: W'] = W
    preds['2: N'] = N
    preds['3: NW'] = NW
    preds['4: N+W-NW'] = N + W - NW
    preds['5: N+(W-NW)/2'] = N + (W - NW) // 2
    preds['6: W+(N-NW)/2'] = W + (N - NW) // 2
    preds['7: (N+W)/2'] = (N + W) // 2
    # New Standard (JPEG-LS / LOCO-I)
    max_WN = np.maximum(W, N)
    min_WN = np.minimum(W, N)
    pred_new = np.where(NW >= max_WN, min_WN,
                        np.where(NW <= min_WN, max_WN,
                                 W + N - NW))
    preds['New (JPEG-LS)'] = pred_new
    return preds


def analyze_image(input_file: str) -> dict:
    """
    Main computation logic. Loads an image file and returns a dictionary with results.
    Args:
        input_file (str): Path to the image file.
    Returns:
        dict: Results with entropy values for original and predictors.
    """
    img = Image.open(input_file).convert('RGB')
    arr = np.array(img, dtype=np.int32)

    stats = {
        "original": {
            "total": calculate_entropy(arr),
            "r": calculate_entropy(arr[:, :, 0]),
            "g": calculate_entropy(arr[:, :, 1]),
            "b": calculate_entropy(arr[:, :, 2])
        },
        "predictors": []
    }

    # Padding for predictors
    padded = np.pad(arr, ((1, 0), (1, 0), (0, 0)), mode='constant', constant_values=0)
    N = padded[:-1, 1:]
    W = padded[1:, :-1]
    NW = padded[:-1, :-1]
    X = arr

    predictors = get_predictors(W, N, NW)
    for name, P in predictors.items():
        diff = (X - P) % 256
        stats["predictors"].append({
            'name': name,
            'total': calculate_entropy(diff),
            'r': calculate_entropy(diff[:, :, 0]),
            'g': calculate_entropy(diff[:, :, 1]),
            'b': calculate_entropy(diff[:, :, 2])
        })
    return stats


def print_results(results: dict, filename: str) -> None:
    """
    Print the results in a readable format.
    Args:
        results (dict): Results dictionary from analyze_image.
        filename (str): Name of the processed file.
    """
    print(f"Processing: {filename}")
    orig = results["original"]
    print("\n--- Original Image ---")
    print(f"Entropy (Total): {orig['total']:.4f}")
    print(f"Entropy (R):     {orig['r']:.4f}")
    print(f"Entropy (G):     {orig['g']:.4f}")
    print(f"Entropy (B):     {orig['b']:.4f}")

    print("\n--- Predictor Results (Error Entropy) ---")
    print(f"{'Method':<20} | {'Total':<8} | {'R':<8} | {'G':<8} | {'B':<8}")
    print("-" * 65)
    for p in results["predictors"]:
        print(f"{p['name']:<20} | {p['total']:.4f}   | {p['r']:.4f}   | {p['g']:.4f}   | {p['b']:.4f}")

    preds = results["predictors"]
    best_total = min(preds, key=lambda x: x['total'])
    best_r = min(preds, key=lambda x: x['r'])
    best_g = min(preds, key=lambda x: x['g'])
    best_b = min(preds, key=lambda x: x['b'])

    print("\n--- Conclusions: Optimal Methods ---")
    print(f"Whole image: {best_total['name']} ({best_total['total']:.4f})")
    print(f"Channel R:   {best_r['name']} ({best_r['r']:.4f})")
    print(f"Channel G:   {best_g['name']} ({best_g['g']:.4f})")
    print(f"Channel B:   {best_b['name']} ({best_b['b']:.4f})")



def main() -> None:
    """
    Main entry point for command-line usage.
    """
    if len(sys.argv) < 2:
        print("Usage: python jpeg_ls.py <image.tga>")
        sys.exit(1)

    input_file = sys.argv[1]
    try:
        results = analyze_image(input_file)
        print_results(results, input_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()