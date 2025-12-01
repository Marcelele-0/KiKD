# tests/test_vq.py
import pytest
import os
import numpy as np
from pathlib import Path
from PIL import Image
from src.vector_quantization import quantize_image

# Ścieżka do obrazków testowych
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
OUTPUT_DIR = Path(__file__).parent.parent / "test_outputs"

@pytest.fixture(autouse=True)
def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_test_images():
    if not TEST_IMAGES_DIR.exists():
        return []
    return list(TEST_IMAGES_DIR.glob("*.tga"))

@pytest.mark.parametrize("image_path", get_test_images())
@pytest.mark.parametrize("bits", [1, 2, 4]) # Testujemy dla 2, 4, 16 kolorów
def test_vq_process(image_path, bits):
    output_path = OUTPUT_DIR / f"{image_path.stem}_{bits}bits.tga"
    
    print(f"Testing {image_path.name} with {bits} bits...")
    
    # Uruchomienie kwantyzacji
    stats = quantize_image(str(image_path), str(output_path), bits)
    
    # 1. Sprawdź czy plik wyjściowy istnieje
    assert output_path.exists()
    
    # 2. Sprawdź poprawność metryk
    assert stats['mse'] >= 0
    # SNR może być inf jeśli obraz jest identyczny (np. mały obrazek, dużo bitów), 
    # ale zazwyczaj jest liczbą dodatnią
    assert stats['snr'] > 0 

    # 3. Sprawdź czy liczba kolorów w pliku wynikowym się zgadza
    # (lub jest mniejsza, jeśli obraz miał mało kolorów)
    out_img = Image.open(output_path)
    out_arr = np.array(out_img)
    # Znajdź unikalne kolory (spłaszczając do listy pikseli)
    unique_colors = np.unique(out_arr.reshape(-1, 3), axis=0)
    
    expected_colors = 2**bits
    assert len(unique_colors) <= expected_colors

def test_vq_zero_bits():
    """Test brzegowy: 0 bitów = 1 kolor (średnia)."""
    # Znajdź dowolny obrazek
    images = get_test_images()
    if not images:
        pytest.skip("Brak obrazków testowych")
        
    img_path = images[0]
    out_path = OUTPUT_DIR / "zero_bits.tga"
    
    stats = quantize_image(str(img_path), str(out_path), 0)
    
    # Obraz powinien mieć tylko 1 unikalny kolor
    out_img = Image.open(out_path)
    out_arr = np.array(out_img)
    unique = np.unique(out_arr.reshape(-1, 3), axis=0)
    
    assert len(unique) == 1