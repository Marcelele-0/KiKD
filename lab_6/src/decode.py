import sys
import pickle
import numpy as np
from PIL import Image
from src.transform import inverse_haar_transform, dpcm_decode
from src.lbg_1d import dequantize_1d

def main() -> None:
    """
    Decodes an image from a custom binary format and saves it as a TGA file.
    Usage: python decode.py <encoded.bin> <output.tga>
    """
    if len(sys.argv) != 3:
        print("Usage: python decode.py <encoded.bin> <output.tga>")
        sys.exit(1)

    input_path: str = sys.argv[1]
    output_path: str = sys.argv[2]

    print(f"Decoding: {input_path} ...")

    # Load encoded data from file
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    width: int = data['width']
    height: int = data['height']
    final_image: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

    for ch in range(3):
        ch_data = data['channels'][ch]

        # 1. Dequantize and reshape to 2D
        flat_low_diff: np.ndarray = dequantize_1d(ch_data['inds_low'], ch_data['cb_low'])
        low_diff: np.ndarray = flat_low_diff.reshape(ch_data['diff_shape'])

        flat_high: np.ndarray = dequantize_1d(ch_data['inds_high'], ch_data['cb_high'])
        high: np.ndarray = flat_high.reshape(ch_data['high_shape'])

        # 2. Inverse DPCM (row-wise)
        low: np.ndarray = dpcm_decode(ch_data['dc_column'], low_diff)

        # 3. Inverse Haar transform
        reconstructed_ch: np.ndarray = inverse_haar_transform(low, high, (height, width))

        final_image[:, :, ch] = reconstructed_ch

    img = Image.fromarray(final_image)
    img.save(output_path)
    print(f"Image saved: {output_path}")


if __name__ == "__main__":
    main()