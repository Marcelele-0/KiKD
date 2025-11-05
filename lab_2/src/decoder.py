import argparse
import time
import logging
from bit_io import BitReader
from adaptive_model import AdaptiveModel
from arithmetic_decoder import ArithmeticDecoder

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger("DECODER_APP")

def main():
    parser = argparse.ArgumentParser(description='Dekodowanie arytmetyczne adaptacyjne.')
    parser.add_argument('input_file', help='Plik wejściowy do zdekompresowania')
    parser.add_argument('output_file', help='Plik wyjściowy (zdekompresowany)')
    args = parser.parse_args()

    log.info(f"Rozpoczynam dekodowanie pliku: {args.input_file}")
    start_time = time.time()

    try:
        with open(args.input_file, "rb") as f_in, open(args.output_file, "wb") as f_out:
            
            model = AdaptiveModel(num_symbols=256)
            reader = BitReader(f_in)
            decoder = ArithmeticDecoder(reader, model)
            
            eof_symbol = model.get_eof_symbol()
            
            while True:
                symbol = decoder.decode_symbol()
                
                if symbol == eof_symbol:
                    log.info("Znaleziono symbol EOF. Koniec dekodowania.")
                    break
                    
                f_out.write(bytes([symbol]))

        end_time = time.time()
        print("\n--- Zakończono dekodowanie ---")
        print(f"Czas wykonania: {end_time - start_time:.3f} s")
        print(f"Plik wyjściowy zapisano jako: {args.output_file}")

    except FileNotFoundError:
        log.error(f"Nie znaleziono pliku {args.input_file}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")
        log.error(f"Wystąpił nieoczekiwany błąd: {e}", exc_info=True)


if __name__ == "__main__":
    main()