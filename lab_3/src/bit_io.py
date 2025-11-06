# bit_io.py

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

    def write_bits_from_string(self, s: str):
        """Pomocnik: zapisuje ciąg bitów (np. "10110")."""
        for bit in s:
            self.write_bit(1 if bit == '1' else 0)

    def flush(self):
        """
        Dopełnia ostatni bajt zerami i zapisuje CAŁY bufor 
        do pliku.
        """
        if self.bit_count > 0:
            padding = 8 - self.bit_count
            self.current_byte <<= padding
            self.main_buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0

        if self.main_buffer:
            self.file.write(self.main_buffer)
            self.main_buffer = bytearray()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

class BitReader:
    """
    Odczytuje bity z pliku, używając dużego bufora (64KB).
    """
    BUFFER_SIZE = 65536 # 64 KB

    def __init__(self, file_object):
        self.file = file_object
        self.buffer = b''
        self.byte_index = 0
        self.bit_index = 8
        self.current_byte = 0
        self.eof = False

    def _load_buffer(self) -> bool:
        """Wczytuje kolejną paczkę 64KB z pliku."""
        self.buffer = self.file.read(self.BUFFER_SIZE)
        self.byte_index = 0
        return len(self.buffer) > 0

    def read_bit(self) -> int:
        """Odczytuje pojedynczy bit. Zwraca -1 jeśli koniec pliku (EOF)."""
        if self.eof:
            return -1

        if self.bit_index == 8:
            if self.byte_index >= len(self.buffer):
                if not self._load_buffer():
                    self.eof = True
                    return -1
            
            self.current_byte = self.buffer[self.byte_index]
            self.byte_index += 1
            self.bit_index = 0

        bit = (self.current_byte >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        
        return bit

    def read_bits(self, count: int):
        """Czyta 'count' bitów i zwraca je jako int. Zwraca (liczba, odczytane_bity)."""
        value = 0
        read_count = 0
        for _ in range(count):
            bit = self.read_bit()
            if bit == -1:
                break
            value = (value << 1) | bit
            read_count += 1
        return value, read_count