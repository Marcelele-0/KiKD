# test/test_lzw.py
import pytest
import sys
import subprocess
import filecmp
from pathlib import Path
import shutil

# --- Definicje Ścieżek ---
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent.parent / "src"
TEST_DATA_DIR = ROOT_DIR / "test_data"
OUTPUT_DIR = ROOT_DIR / "test_outputs"

ENCODER_SCRIPT = SRC_DIR / "lzw_encoder.py"
DECODER_SCRIPT = SRC_DIR / "lzw_decoder.py"
PYTHON_EXE = sys.executable

# --- Przygotowanie danych do testów ---

def get_test_files():
    """Znajduje wszystkie pliki w folderze test_data."""
    try:
        # Dodajemy 'ids' dla ładniejszego wyświetlania w pytest
        files = [p for p in TEST_DATA_DIR.glob('*') if p.is_file()]
        if not files:
            pytest.fail(f"Nie znaleziono żadnych plików testowych w {TEST_DATA_DIR}")
        return files
    except FileNotFoundError:
        pytest.fail(f"Folder test_data nie istnieje: {TEST_DATA_DIR}")

CODERS_TO_TEST = ['omega', 'gamma', 'delta', 'fib']
TEST_FILES = get_test_files()

# --- Fixtures (automatyczne setup/teardown) ---

@pytest.fixture(scope="session")
def setup_output_dirs():
    """
    (Uruchamiane raz na całą sesję testową)
    Tworzy czyste foldery na pliki wyjściowe.
    """
    print(f"\n🧹 Czyszczenie/tworzenie folderu wyjściowego: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    
    for coder in CODERS_TO_TEST:
        (OUTPUT_DIR / "encoded" / coder).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "decoded" / coder).mkdir(parents=True, exist_ok=True)
    print("✅ Foldery gotowe.")
    
    yield
    
    print(f"\n🏁 Testy zakończone. Wyniki w: {OUTPUT_DIR}")


# --- Testy ---

# Dodajemy 'ids' do parametryzacji, aby nazwy testów były czytelne
@pytest.mark.parametrize("original_file_path", TEST_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("coder_name", CODERS_TO_TEST)
def test_lzw_end_to_end(setup_output_dirs, original_file_path, coder_name):
    """
    Testuje pełny cykl kodowania i dekodowania dla danego pliku i kodera.
    """
    file_name = original_file_path.name
    
    sys.path.insert(0, str(SRC_DIR))

    encoded_file_path = OUTPUT_DIR / "encoded" / coder_name / f"{file_name}.lzw"
    decoded_file_path = OUTPUT_DIR / "decoded" / coder_name / f"{file_name}.decoded"

    # 2. Kodowanie
    cmd_encode = [
        PYTHON_EXE, ENCODER_SCRIPT, 
        original_file_path, encoded_file_path, 
        "--coder", coder_name
    ]
    try:
        # --- POPRAWKA: Usunięto text=True i encoding='utf-8' ---
        subprocess.run(cmd_encode, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        # W razie błędu próbujemy zdekodować stderr (z ignorowaniem błędów)
        pytest.fail(f"ENCODE FAILED dla [{file_name} / {coder_name}]\n{e.stderr.decode('utf-8', errors='ignore')}")

    # 3. Dekodowanie
    cmd_decode = [
        PYTHON_EXE, DECODER_SCRIPT, 
        encoded_file_path, decoded_file_path, 
        "--coder", coder_name
    ]
    try:
        # --- POPRAWKA: Usunięto text=True i encoding='utf-8' ---
        subprocess.run(cmd_decode, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"DECODE FAILED dla [{file_name} / {coder_name}]\n{e.stderr.decode('utf-8', errors='ignore')}")

    # 4. Porównanie (asercja)
    assert filecmp.cmp(original_file_path, decoded_file_path, shallow=False), \
           f"Pliki nie są identyczne dla [{file_name} / {coder_name}]"