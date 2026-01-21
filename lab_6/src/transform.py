import numpy as np


def inverse_haar_transform(low: np.ndarray, high: np.ndarray, original_shape: tuple) -> np.ndarray:
def dpcm_encode(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
def dpcm_decode(dc_column: np.ndarray, diffs: np.ndarray) -> np.ndarray:
def apply_haar_transform(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits an image channel into low (average) and high (difference) bands using the Haar transform.
    Operates row-wise.

    Args:
        channel (np.ndarray): 2D array representing a single image channel.

    Returns:
        tuple[np.ndarray, np.ndarray]: (low, high) bands as 2D arrays.
    """
    # Ensure width is even (Haar requires pairs)
    h, w = channel.shape
    if w % 2 != 0:
        channel = channel[:, :-1]
        w -= 1

    channel = channel.astype(float)

    # Slicing: [all rows, every second column]
    even = channel[:, 0::2]
    odd = channel[:, 1::2]

    low = (even + odd) / 2.0
    high = (even - odd) / 2.0

    return low, high

def inverse_haar_transform(low: np.ndarray, high: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """
    Reconstructs a channel from low and high Haar bands.

    Args:
        low (np.ndarray): Low band (average values).
        high (np.ndarray): High band (differences).
        original_shape (tuple[int, int]): Target (height, width) for output.

    Returns:
        np.ndarray: Reconstructed channel as uint8.
    """
    even = low + high
    odd = low - high

    h, w_half = low.shape
    target_h, target_w = original_shape

    # Interleave columns
    reconstructed = np.zeros((h, w_half * 2))
    reconstructed[:, 0::2] = even
    reconstructed[:, 1::2] = odd

    # Handle odd original width (pad last column)
    current_w = reconstructed.shape[1]
    if target_w > current_w:
        # Repeat last column
        last_col = reconstructed[:, -1:]
        reconstructed = np.hstack([reconstructed, last_col])
    elif target_w < current_w:
        reconstructed = reconstructed[:, :target_w]

    return np.clip(reconstructed, 0, 255).astype(np.uint8)

def dpcm_encode(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-wise differential encoding (DPCM).
    Returns:
      - dc_column: First column (start values)
      - diffs: Differences within rows

    Args:
        matrix (np.ndarray): 2D array to encode.

    Returns:
        tuple[np.ndarray, np.ndarray]: (dc_column, diffs)
    """
    # First column is the start points
    dc_column = matrix[:, 0].copy()

    # Differences: col[i] - col[i-1]
    diffs = matrix[:, 1:] - matrix[:, :-1]

    return dc_column, diffs

def dpcm_decode(dc_column: np.ndarray, diffs: np.ndarray) -> np.ndarray:
    """
    Row-wise DPCM decoding.

    Args:
        dc_column (np.ndarray): Start values for each row.
        diffs (np.ndarray): Differences within rows.

    Returns:
        np.ndarray: Decoded 2D array.
    """
    # Concatenate start column with differences
    full_matrix = np.hstack([dc_column.reshape(-1, 1), diffs])

    # Cumulative sum along rows (axis=1)
    return np.cumsum(full_matrix, axis=1)