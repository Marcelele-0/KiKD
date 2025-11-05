import logging
from bit_io import BitWriter
from adaptive_model import AdaptiveModel

logging.basicConfig(level=logging.INFO, format='%(name)10s | %(message)s')
log = logging.getLogger("ENCODER")


class ArithmeticEncoder:
    
    def __init__(self, bit_writer: BitWriter, model: AdaptiveModel, precision: int = 32):
        self.writer = bit_writer
        self.model = model
        
        self.precision = precision
        self.full_range = 1 << self.precision    # << to operacja przesunięcia bitowego w lewo 2^precision
        self.half_range = self.full_range >> 1   # >> to operacja przesunięcia bitowego w prawo o jeden = 2^(precision-1)
        self.quarter_range = self.half_range >> 1 # >> to operacja przesunięcia bitowego w prawo o jeden = 2^(precision-2)
        self.three_q_range = self.half_range + self.quarter_range # 3/4 zakresu
        self.mask = self.full_range - 1 

        self.low = 0
        self.high = self.mask
        self.pending_bits = 0
        log.debug("Encoder zainicjalizowany.")

    def _write_bit_with_pending(self, bit: int):
        self.writer.write_bit(bit)
        log.debug(f"    BIT: Wrote {bit}")
        opposite_bit = 1 - bit
        for _ in range(self.pending_bits):
            self.writer.write_bit(opposite_bit)
            log.debug(f"    BIT: Wrote pending {opposite_bit}")
        self.pending_bits = 0

    def _scale(self):
        while True:
            # E1
            if self.high < self.half_range:
                log.debug(f"    SCALE E1: L={self.low:X}, H={self.high:X}")
                self._write_bit_with_pending(0)
                self.low = (self.low << 1) & self.mask
                self.high = ((self.high << 1) + 1) & self.mask
            
            # E2
            elif self.low >= self.half_range:
                log.debug(f"    SCALE E2: L={self.low:X}, H={self.high:X}")
                self._write_bit_with_pending(1)
                self.low = ((self.low - self.half_range) << 1) & self.mask
                self.high = (((self.high - self.half_range) << 1) + 1) & self.mask
            
            # E3
            elif self.low >= self.quarter_range and self.high < self.three_q_range:
                log.debug(f"    SCALE E3: L={self.low:X}, H={self.high:X} (pending={self.pending_bits})")
                self.pending_bits += 1
                self.low = ((self.low - self.quarter_range) << 1) & self.mask
                self.high = (((self.high - self.quarter_range) << 1) + 1) & self.mask
            
            else:
                break

    def encode_symbol(self, symbol: int):
        char_repr = f"'{chr(symbol)}'" if 32 <= symbol <= 126 else f"[{symbol}]"
        if symbol == self.model.get_eof_symbol(): char_repr = "[EOF]"
        log.debug(f"--- ENCODE symbol={char_repr} ---")
        
        sym_low, sym_high, total = self.model.get_range(symbol) # Pobierz zakres symbolu
        log.debug(f"  Model range: [{sym_low}, {sym_high}) / {total}")

        current_range = self.high - self.low + 1 # 0xFFFFFFFF - 0x00000000 + 1 = 0x100000000 = 2^32
        log.debug(f"  State IN:  L={self.low:X}, H={self.high:X}, R={current_range:X}") 
        
        new_low = self.low + (current_range * sym_low // total)  # operacje na bitach
        new_high = self.low + (current_range * sym_high // total) - 1
        log.debug(f"  State NEW: L={new_low:X}, H={new_high:X}")

        self.low = new_low
        self.high = new_high
        
        self._scale()
        log.debug(f"  State OUT: L={self.low:X}, H={self.high:X}")
        
        self.model.update_model(symbol)

    def finish_encoding(self):
        log.debug("Finishing encoding...")
        self.encode_symbol(self.model.get_eof_symbol())
        
        self.pending_bits += 1
        if self.low < self.quarter_range:
            log.debug("Finish: writing final 0")
            self._write_bit_with_pending(0)
        else:
            log.debug("Finish: writing final 1")
            self._write_bit_with_pending(1)
            
        self.writer.flush()
        log.debug("Encoder finished and flushed.")