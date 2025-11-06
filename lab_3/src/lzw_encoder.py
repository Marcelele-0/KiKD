# lzw_encoder.py
import argparse
import os
import time
from bit_io import BitWriter
from utils import calculate_entropy
from universal_coding import EliasGamma, EliasDelta, EliasOmega, Fibonacci

def get_coder(name: str):
    """Zwraca funkcję kodującą na podstawie nazwy."""
    if name == 'gamma':
        return EliasGamma.encode
    if name == 'delta':
        return EliasDelta.encode
    if name == 'fib':
        return Fibonacci.encode
    if name == 'omega':
        return EliasOmega.encode
    raise ValueError("Nieznany koder")

def main():
    parser = argparse.ArgumentParser(description='Koder LZW z kodowaniem uniwersalnym.')
    parser.add_argument('input_file', help='Plik wejściowy do skompresowania')
    parser.add_argument('output_file', help='Plik wyjściowy (skompresowany)')
    parser.add_argument('--coder', choices=['omega', 'gamma', 'delta', 'fib'],
                        default='omega', help='Typ kodowania uniwersalnego (domyślnie: omega)')
    args = parser.parse_args()

    print(f"Rozpoczynam kodowanie pliku: {args.input_file}")
    print(f"Używane kodowanie uniwersalne: {args.coder}")
    start_time = time.time()

    try:
        # --- Inicjalizacja ---
        encode_function = get_coder(args.coder)
        dictionary = {bytes([i]): i for i in range(256)}
        
        EOF_MARKER = 256 # <-- NOWA ZMIANA 1: Definicja znacznika EOF
        dict_size = 257  # <-- NOWA ZMIANA 2: Nowe wpisy zaczynają się od 257
        
        w = b"" # Aktualnie przetwarzany prefiks
        output_indices = [] # Lista do obliczenia entropii wyjścia

        with open(args.input_file, "rb") as f_in, open(args.output_file, "wb") as f_out:
            writer = BitWriter(f_out)
            input_data = f_in.read()

            # --- Pętla LZW ---
            for byte in input_data:
                k = bytes([byte])
                wk = w + k
                
                if wk in dictionary:
                    w = wk
                else:
                    index = dictionary[w]
                    encode_function(writer, index)
                    output_indices.append(index)
                    
                    dictionary[wk] = dict_size
                    dict_size += 1
                    
                    w = k

            # Zapisz ostatni prefiks 'w'
            if w:
                index = dictionary[w]
                encode_function(writer, index)
                output_indices.append(index)
            
            # --- NOWA ZMIANA 3: Zapisz znacznik EOF ---
            encode_function(writer, EOF_MARKER)
            output_indices.append(EOF_MARKER)
            # --- Koniec zmiany ---

            writer.flush() # Opróżnij bufor bitów

        end_time = time.time()
        
        # --- Obliczanie statystyk ---
        input_size = os.path.getsize(args.input_file)
        output_size = os.path.getsize(args.output_file)
        compression_ratio = input_size / output_size if output_size > 0 else 0
        
        input_entropy = calculate_entropy(list(input_data))
        output_entropy = calculate_entropy(output_indices)

        print("\n--- Zakończono kodowanie ---")
        print(f"Czas wykonania: {end_time - start_time:.3f} s")
        print(f"Długość pliku wejściowego: {input_size} B")
        print(f"Długość kodu wyjściowego: {output_size} B")
        print(f"Stopień kompresji: {compression_ratio:.4f}")
        print("-------------------------------")
        print(f"Entropia tekstu (wejście): {input_entropy:.4f} bitów/symbol")
        print(f"Entropia kodu (indeksy LZW): {output_entropy:.4f} bitów/symbol")

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {args.input_file}")
    except Exception as e:
        print(f"Wystąpił nieoczekiwany błąd: {e}")

if __name__ == "__main__":
    main()