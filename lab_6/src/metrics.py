import numpy as np

def calculate_mse(orig: np.ndarray, recon: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE) between original and reconstructed arrays.

    Args:
        orig (np.ndarray): Original data.
        recon (np.ndarray): Reconstructed data.

    Returns:
        float: Mean squared error value.
    """
    return float(np.mean((orig.astype(float) - recon.astype(float)) ** 2))

def calculate_snr(orig: np.ndarray, mse: float) -> float:
    """
    Calculates Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        orig (np.ndarray): Original data.
        mse (float): Mean squared error value.

    Returns:
        float: SNR value in decibels (dB).
    """
    if mse == 0:
        return float('inf')
    signal_power = np.mean(orig.astype(float) ** 2)
    return 10 * np.log10(signal_power / mse)