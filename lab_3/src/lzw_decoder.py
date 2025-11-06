# lzw_decoder.py
import argparse
import time
from bit_io import BitReader
from universal_coding import EliasGamma, EliasDelta, EliasOmega, Fibonacci

def get_coder(name: str):
    """Zwraca funkcję dekodującą na podstawie nazwy."""
    if name == 'gamma':
        return EliasGamma.decode
    if name == 'delta':
        return EliasDelta.decode
    if name == 'fib':
        return Fibonacci.decode
    if name == 'omega':
        return EliasOmega.decode
    raise ValueError("Nieznany koder")

def main():
    parser = argparse.ArgumentParser(description='Dekoder LZW z kodowaniem uniwersalnym.')
    parser.add_argument('input_file', help='Plik wejściowy do zdekompresowania')
    parser.add_argument('output_file', help='Plik wyjściowy (zdekompresowany)')
    parser.add_argument('--coder', choices=['omega', 'gamma', 'delta', 'fib'],
                        default='omega', help='Typ kodowania uniwersalnego (domyślnie: omega)')
    args = parser.parse_args()

    print(f"Rozpoczynam dekodowanie pliku: {args.input_file}")
    print(f"Używane kodowanie uniwersalne: {args.coder}")
    start_time = time.time()

    try:
        # --- Inicjalizacja ---
        decode_function = get_coder(args.coder)
        
        # --- POPRAWKA BŁĘDU ---
        # Dodajemy pusty wpis (b'') jako "zaślepkę" dla indeksu 256 (EOF_MARKER).
        # Dzięki temu nowe wpisy będą dodawane od indeksu 257, tak jak w koderze.
        dictionary = [bytes([i]) for i in range(256)] + [b'']
        # --- KONIEC POPRAWKI ---
        
        EOF_MARKER = 256
        dict_size = 257

        with open(args.input_file, "rb") as f_in, open(args.output_file, "wb") as f_out:
            reader = BitReader(f_in)
            
            index_old = decode_function(reader)
            if index_old is None or index_old == EOF_MARKER:
                print("Plik wejściowy jest pusty lub uszkodzony.")
                return

            w = dictionary[index_old]
            f_out.write(w)

            # --- Pętla LZW ---
            while True:
                index_new = decode_function(reader)
                
                if index_new is None or index_new == EOF_MARKER:
                    break # Koniec pliku LUB znacznik EOF

                if index_new < dict_size:
                    # Przypadek standardowy
                    entry = dictionary[index_new]
                elif index_new == dict_size:
                    # Przypadek specjalny: KwK...
                    entry = w + w[0:1]
                else:
                    raise ValueError(f"Błąd dekodowania: nieprawidłowy indeks słownika ({index_new})")
                
                f_out.write(entry)
                
                # Słownik jest już wyrównany, append() doda na właściwym indeksie
                dictionary.append(w + entry[0:1])
                dict_size += 1
                
                w = entry

        end_time = time.time()
        print("\n--- Zakończono dekodowanie ---")
        print(f"Czas wykonania: {end_time - start_time:.3f} s")
        print(f"Plik wyjściowy zapisano jako: {args.output_file}")

    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku {args.input_file}")
    except Exception as e:
        print(f"Wystąpił błąd podczas dekodowania: {e}")

if __name__ == "__main__":
    main()