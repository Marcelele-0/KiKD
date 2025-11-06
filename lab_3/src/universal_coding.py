# universal_coding.py
from bit_io import BitWriter, BitReader
import math

# --- Pomocnik dla Kodu Fibonacciego ---
_fib_cache = [1, 2]
def _get_fib_numbers(n: int):
    """Generuje liczby Fibonacciego >= 1 aż przekroczą n."""
    global _fib_cache
    if _fib_cache[-1] > n:
        return [f for f in _fib_cache if f <= n]
    
    while _fib_cache[-1] <= n:
        _fib_cache.append(_fib_cache[-1] + _fib_cache[-2])
    return _fib_cache[:-1]

def _get_fib_decoder_table(max_index=40):
    """Tworzy tabelę dla dekodera."""
    global _fib_cache
    if len(_fib_cache) < max_index:
        while len(_fib_cache) < max_index:
            _fib_cache.append(_fib_cache[-1] + _fib_cache[-2])
    return _fib_cache
# --- Koniec pomocnika ---


class EliasGamma:
    @staticmethod
    def encode(writer: BitWriter, n: int):
        n = n + 1 # Mapowanie n -> n+1 (dla obsługi 0)
        
        unary_len = int(math.log2(n))
        binary_part = bin(n)[3:] # bin(n) daje '0b', [2:] to sama liczba, [3:] to ogon
        
        writer.write_bits_from_string('0' * unary_len + '1' + binary_part)

    @staticmethod
    def decode(reader: BitReader) -> int | None:
        unary_len = 0
        while True:
            bit = reader.read_bit()
            if bit == -1: return None
            if bit == 1: break
            unary_len += 1
        
        val, read_count = reader.read_bits(unary_len)
        if read_count < unary_len: return None # Nieoczekiwany EOF

        n = (1 << unary_len) | val
        return n - 1 # Mapowanie n+1 -> n

class EliasDelta:
    @staticmethod
    def encode(writer: BitWriter, n: int):
        n = n + 1 # Mapowanie n -> n+1
        
        L = int(math.log2(n)) + 1 # Długość binarna n
        binary_part = bin(n)[3:] # Ogon n
        
        EliasGamma.encode(writer, L - 1) # Zakoduj długość (L) używając Gamma
                                         # (L-1), bo Gamma mapuje L -> L+1
        writer.write_bits_from_string(binary_part)

    @staticmethod
    def decode(reader: BitReader) -> int | None:
        L_plus_1 = EliasGamma.decode(reader) # Odczytaj długość (L)
        if L_plus_1 is None: return None
        L = L_plus_1 + 1
        
        val, read_count = reader.read_bits(L - 1)
        if read_count < L - 1: return None # Nieoczekiwany EOF
        
        n = (1 << (L - 1)) | val
        return n - 1 # Mapowanie n+1 -> n

class EliasOmega:
    @staticmethod
    def encode(writer: BitWriter, n: int):
        n = n + 1 # Mapowanie n -> n+1
        
        encoded_string = "0"
        k = n
        while k > 1:
            binary_k = bin(k)[2:] # bin(k) bez '0b'
            encoded_string = binary_k + encoded_string
            k = len(binary_k) - 1
        
        writer.write_bits_from_string(encoded_string)

    @staticmethod
    def decode(reader: BitReader) -> int | None:
        k = 1
        while True:
            first_bit = reader.read_bit()
            if first_bit == -1: return None
            if first_bit == 0:
                return k - 1 # Mapowanie n+1 -> n
            
            # Czytaj 'k' kolejnych bitów (poza pierwszym '1')
            val, read_count = reader.read_bits(k)
            if read_count < k: return None # Nieoczekiwany EOF
            
            k = (1 << k) | val

class Fibonacci:
    # Używamy statycznej tabeli do dekodowania, wystarczająco dużej
    F = _get_fib_decoder_table(40)

    @staticmethod
    def encode(writer: BitWriter, n: int):
        n = n + 1 # Mapowanie n -> n+1
        
        fib_nums = _get_fib_numbers(n)
        index = len(fib_nums) - 1
        codeword = ['0'] * (index + 1)
        
        temp_n = n
        while temp_n > 0:
            while fib_nums[index] > temp_n:
                index -= 1
            
            codeword[index] = '1'
            temp_n -= fib_nums[index]
            index -= 1
            
        writer.write_bits_from_string("".join(codeword) + "1") # Dodaj '1' na końcu

    @staticmethod
    def decode(reader: BitReader) -> int | None:
        n = 0
        last_bit = -1
        index = 0
        F = Fibonacci.F # Użyj statycznej tabeli
        
        while True:
            bit = reader.read_bit()
            if bit == -1: return None
            
            if last_bit == 1 and bit == 1:
                # Znaleziono "11" - koniec kodu
                return n - 1 # Mapowanie n+1 -> n
            
            if bit == 1:
                if index >= len(F):
                    raise OverflowError("Błąd dekodowania Fibonacciego: indeks poza tabelą")
                n += F[index]
            
            last_bit = bit
            index += 1