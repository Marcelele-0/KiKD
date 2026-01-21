import pytest
import numpy as np
import os
from pathlib import Path
from PIL import Image

from src.transform import (
    apply_haar_transform,
    inverse_haar_transform,
    dpcm_encode,
    dpcm_decode,
)
from src.lbg_1d import train_lbg_1d, quantize_1d, dequantize_1d
from src.metrics import calculate_mse, calculate_snr

# Paths for test images and outputs
TEST_IMAGES_DIR: Path = Path(__file__).parent.parent / "test_images"
OUTPUT_DIR: Path = Path(__file__).parent.parent / "test_outputs"

@pytest.fixture(autouse=True)
def setup_dirs() -> None:
    """Ensure output directory exists before tests run."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_test_images() -> list[Path]:
    """Return a list of test image paths from the test_images directory."""
    if not TEST_IMAGES_DIR.exists():
        return []
    extensions = ["*.tga", "*.png", "*.jpg"]
    files: list[Path] = []
    for ext in extensions:
        files.extend(TEST_IMAGES_DIR.glob(ext))
    return sorted(list(set(files)))

@pytest.mark.parametrize("image_path", get_test_images())
@pytest.mark.parametrize("k_bits", [3, 5])
def test_full_report(image_path: Path, k_bits: int) -> None:
    """
    Full encode-decode test for an image, reporting MSE and SNR for each channel and total.
    Saves the reconstructed image to the output directory.
    """
    print(f"\n{'='*60}")
    print(f"TEST: {image_path.name} | k_bits = {k_bits}")
    print(f"{'='*60}")

    # Load image
    img = Image.open(image_path).convert("RGB")
    arr_orig: np.ndarray = np.array(img)
    height, width, _ = arr_orig.shape

    # Handle odd width for fair MSE calculation
    width_proc: int = width if width % 2 == 0 else width - 1
    arr_recon: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
    arr_orig_cropped: np.ndarray = arr_orig[:, :width_proc, :]

    # Loop over channels (simulate encode -> decode)
    channel_names = ["Red", "Green", "Blue"]
    for ch in range(3):
        channel: np.ndarray = arr_orig[:, :, ch]

        # ENCODE
        low, high = apply_haar_transform(channel)
        dc_col, low_diff = dpcm_encode(low)

        flat_low = low_diff.flatten()
        flat_high = high.flatten()

        cb_low = train_lbg_1d(flat_low[::5], k_bits)
        cb_high = train_lbg_1d(flat_high[::5], k_bits)

        inds_low = quantize_1d(flat_low, cb_low)
        inds_high = quantize_1d(flat_high, cb_high)

        # DECODE
        rec_flat_low = dequantize_1d(inds_low, cb_low)
        rec_flat_high = dequantize_1d(inds_high, cb_high)

        rec_low_diff = rec_flat_low.reshape(low_diff.shape)
        rec_high = rec_flat_high.reshape(high.shape)

        rec_low = dpcm_decode(dc_col, rec_low_diff)
        rec_ch = inverse_haar_transform(rec_low, rec_high, (height, width))

        arr_recon[:, :, ch] = rec_ch

    # Save reconstructed image
    out_name = f"{image_path.stem}_k{k_bits}.tga"
    out_path = OUTPUT_DIR / out_name
    Image.fromarray(arr_recon).save(out_path)
    print(f"[SAVED] {out_name}")

    # Metrics
    print("\n[RESULTS]")
    print(f"{'Channel':<10} | {'MSE':<10} | {'SNR (dB)':<10}")
    print("-" * 36)

    # Calculate only on common part (width_proc)
    arr_recon_cropped = arr_recon[:, :width_proc, :]

    mse_total = calculate_mse(arr_orig_cropped, arr_recon_cropped)
    snr_total = calculate_snr(arr_orig_cropped, mse_total)

    for ch in range(3):
        mse_c = calculate_mse(arr_orig_cropped[:,:,ch], arr_recon_cropped[:,:,ch])
        snr_c = calculate_snr(arr_orig_cropped[:,:,ch], mse_c)
        print(f"{channel_names[ch]:<10} | {mse_c:<10.4f} | {snr_c:<10.4f}")

    print("-" * 36)
    print(f"{'TOTAL':<10} | {mse_total:<10.4f} | {snr_total:<10.4f}")

    assert out_path.exists()