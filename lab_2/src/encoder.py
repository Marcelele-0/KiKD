import argparse
import os
import math
import time
import logging
from typing import Tuple
from collections import Counter
from bit_io import BitWriter
from adaptive_model import AdaptiveModel
from arithmetic_encoder import ArithmeticEncoder

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("ENCODER_APP")

def calculate_entropy(data_bytes: bytes) -> Tuple[float, int]:
    """Oblicza entropię (H) danych wejściowych."""
    if not data_bytes:
        return 0.0, 0
        
    counts = Counter(data_bytes)
    total_bytes = len(data_bytes)
    entropy = 0.0
    
    for count in counts.values():
        probability = count / total_bytes
        entropy -= probability * math.log2(probability)
        
    return entropy, total_bytes

def main():
    parser = argparse.ArgumentParser(description='Kodowanie arytmetyczne adaptacyjne.')
    parser.add_argument('input_file', help='Plik wejściowy do skompresowania')
    parser.add_argument('output_file', help='Plik wyjściowy (skompresowany)')
    args = parser.parse_args()

    log.info(f"Rozpoczynam kodowanie pliku: {args.input_file}")
    start_time = time.time()

    try:
        with open(args.input_file, "rb") as f_in, open(args.output_file, "wb") as f_out:
            input_data = f_in.read()
            
            model = AdaptiveModel(num_symbols=256)
            writer = BitWriter(f_out)
            encoder = ArithmeticEncoder(writer, model)

            for byte in input_data:
                encoder.encode_symbol(byte)
            
            encoder.finish_encoding()

        end_time = time.time()
        
        entropy, total_bytes = calculate_entropy(input_data)
        input_size = os.path.getsize(args.input_file)
        output_size = os.path.getsize(args.output_file)
        
        avg_length = 0.0
        if total_bytes > 0:
            avg_length = (output_size * 8) / total_bytes
            
        compression_ratio = 0.0
        if output_size > 0:
            compression_ratio = input_size / output_size

        print("\n--- Zakończono kodowanie ---")
        print(f"Czas wykonania: {end_time - start_time:.3f} s")
        print(f"Rozmiar wejściowy: {input_size} B")
        print(f"Rozmiar wyjściowy: {output_size} B")
        print("-------------------------------")
        print(f"Entropia danych (H):   {entropy:.4f} bitów/symbol")
        print(f"Śr. dł. kodowania (L): {avg_length:.4f} bitów/symbol")
        print(f"Stopień kompresji:     {compression_ratio:.4f}")

    except FileNotFoundError:
        log.error(f"Nie znaleziono pliku {args.input_file}")
    except Exception as e:
        log.error(f"Wystąpił nieoczekiwany błąd: {e}", exc_info=True)


if __name__ == "__main__":
    main()