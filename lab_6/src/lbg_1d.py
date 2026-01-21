import numpy as np


def quantize_1d(data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
def dequantize_1d(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
def train_lbg_1d(data: np.ndarray, num_bits: int, epsilon: float = 0.01) -> np.ndarray:
    """
    Trains a non-uniform quantizer (LBG algorithm) for 1D data.
    Returns a sorted codebook.

    Args:
        data (np.ndarray): Input data to quantize.
        num_bits (int): Number of bits for quantization (codebook size = 2**num_bits).
        epsilon (float): Small value for codebook splitting.

    Returns:
        np.ndarray: Sorted codebook (centroids).
    """
    target_levels = 2 ** num_bits

    if len(data) == 0:
        return np.array([0.0])

    # Start: mean of all data
    codebook = np.array([np.mean(data)])

    # Splitting loop
    while len(codebook) < target_levels:
        lower = codebook * (1 - epsilon)
        upper = codebook * (1 + epsilon)
        codebook = np.sort(np.concatenate([lower, upper]))

        # Optimization (1D K-Means) - 10 iterations
        for _ in range(10):
            # Assign to nearest centroid
            distances = np.abs(data.reshape(-1, 1) - codebook.reshape(1, -1))
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_codebook = np.zeros_like(codebook)
            for i in range(len(codebook)):
                cluster = data[labels == i]
                if len(cluster) > 0:
                    new_codebook[i] = np.mean(cluster)
                else:
                    new_codebook[i] = codebook[i]
            codebook = new_codebook

    return np.sort(codebook)

def quantize_1d(data: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    Quantizes values to codebook indices.

    Args:
        data (np.ndarray): Input data to quantize.
        codebook (np.ndarray): Codebook (centroids).

    Returns:
        np.ndarray: Indices of closest codebook values.
    """
    distances = np.abs(data.reshape(-1, 1) - codebook.reshape(1, -1))
    return np.argmin(distances, axis=1).astype(np.uint16)

def dequantize_1d(indices: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    Dequantizes indices to values using the codebook.

    Args:
        indices (np.ndarray): Indices to decode.
        codebook (np.ndarray): Codebook (centroids).

    Returns:
        np.ndarray: Decoded values.
    """
    return codebook[indices]