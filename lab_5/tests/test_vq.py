# tests/test_rgb_variants.py
import pytest
import os
from pathlib import Path
from PIL import Image

# Konfiguracja ścieżek
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
OUTPUT_DIR = Path(__file__).parent.parent / "test_outputs" / "rgb_variants"

@pytest.fixture(autouse=True)
def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_test_images():
    if not TEST_IMAGES_DIR.exists():
        return []
    # Szukamy obrazka z twarzą (lub pierwszego dostępnego)
    files = list(TEST_IMAGES_DIR.glob("*.png")) + list(TEST_IMAGES_DIR.glob("*.tga"))
    return files

# Lista wariantów z Twojego obrazka
VARIANTS = ["000", "800", "080", "008", "088", "808", "880", "888"]

def apply_channel_mask(img, mode_str):
    """
    mode_str: np. "800" -> R=8bit, G=0bit, B=0bit
    """
    # Rozdziel obraz na kanały R, G, B
    channels = img.split()
    if len(channels) > 3:
        channels = channels[:3] # Ignoruj Alpha jeśli jest
    
    r, g, b = channels
    
    # Jeśli w stringu jest '0', zamieniamy kanał na czarny (zero bitów)
    # Jeśli jest '8', zostawiamy bez zmian (8 bitów)
    if mode_str[0] == '0':
        r = r.point(lambda _: 0)
    
    if mode_str[1] == '0':
        g = g.point(lambda _: 0)
        
    if mode_str[2] == '0':
        b = b.point(lambda _: 0)
        
    # Złącz kanały z powrotem
    return Image.merge("RGB", (r, g, b))

@pytest.mark.parametrize("image_path", get_test_images())
@pytest.mark.parametrize("mode", VARIANTS)
def test_generate_rgb_variant(image_path, mode):
    """
    Generuje warianty obrazu: 800 (tylko czerwony), 080 (tylko zielony) itd.
    """
    output_path = OUTPUT_DIR / f"{image_path.stem}_{mode}.png"
    
    print(f"Generating {mode} variant for {image_path.name}...")
    
    img = Image.open(image_path).convert("RGB")
    
    # Zastosuj maskowanie kanałów
    new_img = apply_channel_mask(img, mode)
    
    new_img.save(output_path)
    
    # Weryfikacja: Plik powstał
    assert output_path.exists()
    
    # Weryfikacja treści (np. dla 800 kanał G i B muszą być czarne)
    data = new_img.getdata()
    sample_pixel = data[0] # Pobierz pierwszy piksel
    
    if mode[0] == '0': assert sample_pixel[0] == 0 # Red powinien być 0
    if mode[1] == '0': assert sample_pixel[1] == 0 # Green powinien być 0
    if mode[2] == '0': assert sample_pixel[2] == 0 # Blue powinien być 0