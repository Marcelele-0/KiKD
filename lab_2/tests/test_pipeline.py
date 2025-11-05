import unittest
import tempfile
import os
import sys
from pathlib import Path

# Dodaj src do path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bit_io import BitWriter, BitReader
from adaptive_model import AdaptiveModel
from arithmetic_encoder import ArithmeticEncoder
from arithmetic_decoder import ArithmeticDecoder


class TestEncoderDecoderPipeline(unittest.TestCase):
    """Testy sprawdzające cały pipeline encoding/decoding"""

    def setUp(self):
        """Przygotowanie do każdego testu"""
        self.project_root = Path(__file__).parent.parent
        self.test_data_dir = self.project_root / "test_data"
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Czyszczenie plików tymczasowych"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _encode_file(self, input_path: str, output_path: str):
        """Koduje plik"""
        with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            data = f_in.read()
            model = AdaptiveModel(num_symbols=256)
            writer = BitWriter(f_out)
            encoder = ArithmeticEncoder(writer, model)

            for byte in data:
                encoder.encode_symbol(byte)
            encoder.finish_encoding()

    def _decode_file(self, input_path: str, output_path: str):
        """Dekoduje plik"""
        with open(input_path, "rb") as f_in, open(output_path, "wb") as f_out:
            model = AdaptiveModel(num_symbols=256)
            reader = BitReader(f_in)
            decoder = ArithmeticDecoder(reader, model)

            eof_symbol = model.get_eof_symbol()

            while True:
                symbol = decoder.decode_symbol()
                if symbol == eof_symbol:
                    break
                f_out.write(bytes([symbol]))

    def _test_file_roundtrip(self, test_file_path: str):
        """Koduje i dekoduje plik, porównuje z oryginałem"""
        # Przeczytaj oryginalny plik
        with open(test_file_path, "rb") as f:
            original_data = f.read()

        # Koduj
        encoded_path = os.path.join(self.temp_dir, "encoded.bin")
        self._encode_file(test_file_path, encoded_path)
        self.assertTrue(os.path.exists(encoded_path), "Plik zakodowany nie został utworzony")

        # Dekoduj
        decoded_path = os.path.join(self.temp_dir, "decoded.bin")
        self._decode_file(encoded_path, decoded_path)
        self.assertTrue(os.path.exists(decoded_path), "Plik zdekodowany nie został utworzony")

        # Porównaj
        with open(decoded_path, "rb") as f:
            decoded_data = f.read()

        self.assertEqual(
            original_data,
            decoded_data,
            f"Dane zdekodowane różnią się od oryginału!\nOryginalny rozmiar: {len(original_data)}, "
            f"zdekodowany rozmiar: {len(decoded_data)}"
        )

        # Wyświetl statystyki
        encoded_size = os.path.getsize(encoded_path)
        compression_ratio = len(original_data) / encoded_size if encoded_size > 0 else 0
        print(f"\n✓ {os.path.basename(test_file_path)}: "
              f"oryg={len(original_data)}B, "
              f"kodow={encoded_size}B, "
              f"ratio={compression_ratio:.2f}")

    def test_pan_txt(self):
        """Test na pliku pan.txt"""
        test_file = self.test_data_dir / "pan.txt"
        if test_file.exists():
            self._test_file_roundtrip(str(test_file))
        else:
            self.skipTest(f"Plik {test_file} nie istnieje")

    def test_test1_bin(self):
        """Test na pliku test1.bin"""
        test_file = self.test_data_dir / "test1.bin"
        if test_file.exists():
            self._test_file_roundtrip(str(test_file))
        else:
            self.skipTest(f"Plik {test_file} nie istnieje")

    def test_test2_bin(self):
        """Test na pliku test2.bin"""
        test_file = self.test_data_dir / "test2.bin"
        if test_file.exists():
            self._test_file_roundtrip(str(test_file))
        else:
            self.skipTest(f"Plik {test_file} nie istnieje")

    def test_test3_bin(self):
        """Test na pliku test3.bin"""
        test_file = self.test_data_dir / "test3.bin"
        if test_file.exists():
            self._test_file_roundtrip(str(test_file))
        else:
            self.skipTest(f"Plik {test_file} nie istnieje")

    def test_empty_file(self):
        """Test na pustym pliku"""
        empty_file = os.path.join(self.temp_dir, "empty.txt")
        with open(empty_file, "wb") as f:
            pass  # Utwórz pusty plik

        self._test_file_roundtrip(empty_file)

    def test_single_byte(self):
        """Test na jednobajtowym pliku"""
        single_byte_file = os.path.join(self.temp_dir, "single.bin")
        with open(single_byte_file, "wb") as f:
            f.write(b"\x42")

        self._test_file_roundtrip(single_byte_file)

    def test_all_byte_values(self):
        """Test na pliku zawierającym wszystkie możliwe wartości bajtów"""
        all_bytes_file = os.path.join(self.temp_dir, "all_bytes.bin")
        with open(all_bytes_file, "wb") as f:
            f.write(bytes(range(256)))

        self._test_file_roundtrip(all_bytes_file)

    def test_repeated_pattern(self):
        """Test na pliku z powtarzającym się wzorem"""
        pattern_file = os.path.join(self.temp_dir, "pattern.bin")
        with open(pattern_file, "wb") as f:
            pattern = b"ABCDEFG"
            f.write(pattern * 1000)

        self._test_file_roundtrip(pattern_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
