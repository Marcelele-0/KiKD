import pytest
import os
import numpy as np
from pathlib import Path
from PIL import Image
from src.vector_quantization import quantize_image

SRC_DIR = Path(__file__).parent.parent / "src"
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
OUTPUT_DIR = Path(__file__).parent.parent / "test_outputs"

@pytest.fixture(autouse=True)
def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_test_images():
    if not TEST_IMAGES_DIR.exists():
        return []
    files = list(TEST_IMAGES_DIR.glob("*.tga")) + list(TEST_IMAGES_DIR.glob("*.png"))
    return sorted(files)

@pytest.mark.parametrize("image_path", get_test_images())
@pytest.mark.parametrize("k", [2, 3, 12]) 
def test_vq_process(image_path, k):
    output_path = OUTPUT_DIR / f"{image_path.stem}_k{k}.tga"
    
    print(f"\nTesting {image_path.name} with K={k} ({2**k} colors)...")
    
    stats = quantize_image(str(image_path), str(output_path), k)
    
    # --- DODANE PRINTY ---
    print(f"  MSE: {stats['mse']:.4f}")
    print(f"  SNR: {stats['snr']:.4f} dB")
    # ---------------------
    
    assert output_path.exists()
    assert stats['mse'] >= 0
    assert stats['snr'] > 0

    out_img = Image.open(output_path)
    out_arr = np.array(out_img)
    unique_colors = np.unique(out_arr.reshape(-1, 3), axis=0)
    
    expected_max = 2**k
    
    print(f"  Colors found: {len(unique_colors)} (Max allowed: {expected_max})")
    assert len(unique_colors) <= expected_max

def test_vq_zero_k():
    images = get_test_images()
    if not images:
        pytest.skip("Brak obrazów")
        
    img_path = images[0]
    out_path = OUTPUT_DIR / "test_k0.tga"
    
    quantize_image(str(img_path), str(out_path), 0)
    
    out_img = Image.open(out_path)
    unique = np.unique(np.array(out_img).reshape(-1, 3), axis=0)
    
    assert len(unique) == 1