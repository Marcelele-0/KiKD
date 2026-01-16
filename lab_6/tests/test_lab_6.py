import pytest
import numpy as np
import os
from pathlib import Path
from PIL import Image

# Importujemy funkcje bezpośrednio z Twoich modułów src
from src.transform import (
    apply_haar_transform,
    inverse_haar_transform,
    dpcm_encode,
    dpcm_decode,
)
from src.lbg_1d import train_lbg_1d, quantize_1d, dequantize_1d
from src.metrics import calculate_mse, calculate_snr

# Konfiguracja ścieżek
# Zakładamy, że test jest w folderze tests/, a obrazy w test_images/ obok src/
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"
OUTPUT_DIR = Path(__file__).parent.parent / "test_outputs"


@pytest.fixture(autouse=True)
def setup_dirs():
    """Upewnia się, że katalog wyjściowy istnieje."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_test_images() -> list[Path]:
    """Zwraca listę ścieżek do obrazów testowych (TGA, PNG, JPG)."""
    if not TEST_IMAGES_DIR.exists():
        return []
    
    extensions = ["*.tga", "*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(TEST_IMAGES_DIR.glob(ext))
    
    # Sortujemy dla powtarzalności kolejności testów
    return sorted(list(set(files)))


@pytest.mark.parametrize("image_path", get_test_images())
@pytest.mark.parametrize("k_bits", [2, 5])  # Testujemy dla k=2 (4 poziomy) i k=5 (32 poziomy)
def test_compression_pipeline(image_path: Path, k_bits: int):
    """
    Testuje pełny potok: Transformata -> DPCM -> LBG -> Kwantyzacja -> Rekonstrukcja.
    Weryfikuje, czy algorytmy na ocenę 5 działają poprawnie razem.
    """
    print(f"\nTesting {image_path.name} with K={k_bits}...")

    # 1. Wczytanie obrazu
    img = Image.open(image_path).convert("RGB")
    arr_orig = np.array(img)
    height, width, _ = arr_orig.shape
    
    # Tablica na wynik rekonstrukcji
    # Uwaga: Transformata Haara w naszej implementacji może uciąć 1 piksel, jeśli szerokość jest nieparzysta
    # Dlatego przygotowujemy wynik o wymiarach "procesowanych"
    width_proc = width if width % 2 == 0 else width - 1
    arr_recon = np.zeros((height, width_proc, 3), dtype=np.uint8)

    # 2. Przetwarzanie kanałów (Symulacja encode.py + decode.py w pamięci)
    for ch in range(3):
        channel = arr_orig[:, :, ch]

        # --- ENCODING ---
        
        # A. Transformata (Sub-band)
        low, high = apply_haar_transform(channel)
        
        # B. DPCM na paśmie Low
        low_diff = dpcm_encode(low)
        
        # C. Trening LBG (Codebooki)
        # Używamy próbki danych dla szybkości testu, tak jak w skrypcie
        cb_low = train_lbg_1d(low_diff[::5], k_bits)
        cb_high = train_lbg_1d(high[::5], k_bits)
        
        # Asercja: Czy codebook ma poprawny rozmiar? (2^k)
        assert len(cb_low) == 2**k_bits
        assert len(cb_high) == 2**k_bits
        # Asercja: Czy codebook jest posortowany?
        assert np.all(cb_low[:-1] <= cb_low[1:])

        # D. Kwantyzacja (Wartości -> Indeksy)
        inds_low = quantize_1d(low_diff, cb_low)
        inds_high = quantize_1d(high, cb_high)
        
        # --- DECODING ---

        # E. Dekwantyzacja (Indeksy -> Wartości)
        rec_low_diff = dequantize_1d(inds_low, cb_low)
        rec_high = dequantize_1d(inds_high, cb_high)

        # F. Odwrócenie DPCM
        rec_low = dpcm_decode(rec_low_diff)

        # G. Odwrócenie Transformaty
        rec_channel = inverse_haar_transform(rec_low, rec_high, (height, width))
        
        # Zapis do tablicy wynikowej (clipowanie wartości)
        arr_recon[:, :, ch] = np.clip(rec_channel, 0, 255).astype(np.uint8)

    # 3. Zapisz wynik testu (opcjonalnie, żebyś mógł podejrzeć)
    out_path = OUTPUT_DIR / f"test_result_{image_path.stem}_k{k_bits}.tga"
    Image.fromarray(arr_recon).save(out_path)

    # 4. Weryfikacja Metryk
    # Musimy przyciąć oryginał do szerokości rekonstrukcji (jeśli był nieparzysty)
    arr_orig_cropped = arr_orig[:, :width_proc, :]
    
    mse = calculate_mse(arr_orig_cropped, arr_recon)
    snr = calculate_snr(arr_orig_cropped, mse)

    print(f"  -> MSE: {mse:.4f}, SNR: {snr:.4f} dB")

    # Asercje końcowe
    assert mse >= 0
    # SNR powinien być rozsądny (chyba że obraz jest czarny/pusty). 
    # Dla k=5 zazwyczaj SNR > 20dB dla zdjęć.
    if np.mean(arr_orig) > 10: 
        assert snr > 5.0 


def test_lbg_convergence():
    """
    Test jednostkowy sprawdzający, czy algorytm LBG faktycznie redukuje błąd kwantyzacji
    w porównaniu do zwykłego średniego.
    """
    # Generujemy dane: dwa skupiska (np. okolice 0 i okolice 100)
    data = np.concatenate([np.random.normal(0, 5, 1000), np.random.normal(100, 5, 1000)])
    
    # Uczymy LBG na 1 bit (powinien znaleźć 2 centra: ok. 0 i ok. 100)
    cb = train_lbg_1d(data, num_bits=1)
    
    assert len(cb) == 2
    # Sprawdź czy centra są w miarę sensowne (blisko 0 i 100)
    assert -10 < cb[0] < 10
    assert 90 < cb[1] < 110


def test_transform_invertibility():
    """
    Sprawdza, czy transformata Haara i DPCM są odwracalne (bez kwantyzacji).
    To zapewnia, że strata jakości pochodzi TYLKO z kwantyzacji.
    """
    # Losowe dane (kanał obrazu)
    original = np.random.randint(0, 255, (100, 100)).astype(float)
    
    # 1. Transformacja w przód
    L, H = apply_haar_transform(original)
    L_diff = dpcm_encode(L)
    
    # 2. Transformacja w tył (bez kwantyzacji!)
    L_rec = dpcm_decode(L_diff)
    
    # Uwaga: inverse_haar oczekuje, że znamy oryginalny kształt
    final = inverse_haar_transform(L_rec, H, original.shape)
    
    # Sprawdź czy odzyskaliśmy oryginał (z dokładnością do float epsilon)
    assert np.allclose(original, final, atol=1e-5)