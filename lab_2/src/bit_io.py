class BitWriter:
    """
    Zapisuje bity do bufora w pamięci (bytearray), a na końcu
    zapisuje cały bufor do pliku jednym wywołaniem write().
    """
    def __init__(self, file_object):
        self.file = file_object
        self.main_buffer = bytearray()
        self.current_byte = 0
        self.bit_count = 0

    def write_bit(self, bit: int):
        """Zapisuje pojedynczy bit (0 lub 1) do bufora bajtu."""
        self.current_byte = (self.current_byte << 1) | bit
        self.bit_count += 1
        
        if self.bit_count == 8:
            self.main_buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0

    def flush(self):
        """
        Dopełnia ostatni bajt zerami i zapisuje CAŁY bufor 
        do pliku.
        """
        if self.bit_count > 0:
            # Dopełnij zerami z prawej strony
            padding = 8 - self.bit_count
            self.current_byte <<= padding
            self.main_buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0

        # Wykonaj tylko JEDNĄ operację zapisu do pliku
        if self.main_buffer:
            self.file.write(self.main_buffer)
            self.main_buffer = bytearray() # Wyczyść bufor

    def __enter__(self):
        return self

    def __exit__(self):
        # Automatycznie wywołaj flush() przy wyjściu z bloku 'with'
        self.flush()

class BitReader:
    """
    Odczytuje bity z pliku, używając dużego bufora (64KB),
    aby zminimalizować liczbę wywołań systemowych read().
    """
    BUFFER_SIZE = 65536 # 64 KB

    def __init__(self, file_object):
        self.file = file_object
        self.buffer = b''          # Bufor na bajty wczytane z pliku
        self.byte_index = 0        # Indeks w self.buffer
        self.bit_index = 8         # Indeks bitu w self.current_byte (0-7)
                                   # Zaczyna od 8, by wymusić wczytanie przy 1. wywołaniu
        self.current_byte = 0

    def _load_buffer(self) -> bool:
        """Wczytuje kolejną paczkę 64KB z pliku."""
        self.buffer = self.file.read(self.BUFFER_SIZE)
        self.byte_index = 0
        return len(self.buffer) > 0

    def read_bit(self) -> int:
        """Odczytuje pojedynczy bit. Zwraca -1 jeśli koniec pliku (EOF)."""
        
        if self.bit_index == 8: # Potrzebujemy nowego bajtu
            if self.byte_index >= len(self.buffer):
                # Potrzebujemy nowego bufora
                if not self._load_buffer():
                    return -1  # Koniec pliku
            
            # Pobierz nowy bajt z bufora
            self.current_byte = self.buffer[self.byte_index]
            self.byte_index += 1
            self.bit_index = 0 # Resetuj indeks bitu

        # Odczytaj bit (od MSB do LSB)
        # bit_index 0 -> (7-0) = bit 7 (MSB)
        # bit_index 7 -> (7-7) = bit 0 (LSB)
        bit = (self.current_byte >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        
        return bit