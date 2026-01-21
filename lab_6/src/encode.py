import sys
import pickle
import numpy as np
from PIL import Image
from src.transform import apply_haar_transform, dpcm_encode
from src.lbg_1d import train_lbg_1d, quantize_1d

def main() -> None:
    """
    Encodes an image to a custom binary format using Haar transform, DPCM, and LBG quantization.
    Usage: python encode.py <input.tga> <output.bin> <k_bits>
    """
    if len(sys.argv) != 4:
        print("Usage: python encode.py <input.tga> <output.bin> <k_bits>")
        sys.exit(1)

    input_path: str = sys.argv[1]
    output_path: str = sys.argv[2]

    try:
        k_bits: int = int(sys.argv[3])
        if not (1 <= k_bits <= 7):
            raise ValueError("k_bits must be in range 1-7")
    except ValueError as e:
        print(f"Argument error: {e}")
        sys.exit(1)

    print(f"Encoding: {input_path} (k_bits={k_bits}) ...")
    img = Image.open(input_path).convert('RGB')
    arr: np.ndarray = np.array(img)
    height, width, _ = arr.shape

    encoded_data: dict = {
        'width': width,
        'height': height,
        'k': k_bits,
        'channels': []
    }

    for ch in range(3):
        channel_pixels: np.ndarray = arr[:, :, ch]

        # 1. Haar 2D transform
        low, high = apply_haar_transform(channel_pixels)

        # 2. DPCM on rows (only Low band)
        dc_column, low_diff = dpcm_encode(low)

        # 3. LBG training (flatten matrices for histogram learning)
        # Sample every 10th pixel for speed
        print(f"  Channel {ch}: LBG training ...")
        cb_low = train_lbg_1d(low_diff.flatten()[::10], k_bits)
        cb_high = train_lbg_1d(high.flatten()[::10], k_bits)

        # 4. Quantization (Values -> Indices)
        inds_low = quantize_1d(low_diff.flatten(), cb_low)
        inds_high = quantize_1d(high.flatten(), cb_high)

        # Store matrix shapes for decoder to reshape
        encoded_data['channels'].append({
            'dc_column': dc_column,   # DPCM start vector
            'diff_shape': low_diff.shape,
            'high_shape': high.shape,
            'cb_low': cb_low,
            'cb_high': cb_high,
            'inds_low': inds_low,
            'inds_high': inds_high
        })

    # Save encoded data to file
    with open(output_path, 'wb') as f:
        pickle.dump(encoded_data, f)

    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()