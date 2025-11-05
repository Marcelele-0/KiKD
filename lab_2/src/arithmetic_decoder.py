import logging
from bit_io import BitReader
from adaptive_model import AdaptiveModel

logging.basicConfig(level=logging.INFO, format='%(name)10s | %(message)s')
log = logging.getLogger("DECODER")


class ArithmeticDecoder:
    
    def __init__(self, bit_reader: BitReader, model: AdaptiveModel, precision: int = 32):
        self.reader = bit_reader
        self.model = model
        
        self.precision = precision
        self.full_range = 1 << self.precision
        self.half_range = self.full_range >> 1
        self.quarter_range = self.half_range >> 1
        self.three_q_range = self.half_range + self.quarter_range
        self.mask = self.full_range - 1

        self.low = 0
        self.high = self.mask
        
        self.value = 0
        for i in range(self.precision):
            bit = self._read_next_bit()
            self.value = ((self.value << 1) | bit) & self.mask
        
        log.debug("Decoder zainicjalizowany.")
        log.debug(f"Initial Value: V={self.value:X}")

    def _scale(self):
        while True:
            # E1
            if self.high < self.half_range:
                log.debug(f"    SCALE E1: L={self.low:X}, H={self.high:X}")
                self.low = (self.low << 1) & self.mask
                self.high = ((self.high << 1) + 1) & self.mask
                self.value = ((self.value << 1) | self._read_next_bit()) & self.mask
                log.debug(f"      -> V={self.value:X}")
            
            # E2
            elif self.low >= self.half_range:
                log.debug(f"    SCALE E2: L={self.low:X}, H={self.high:X}")
                self.low = ((self.low - self.half_range) << 1) & self.mask
                self.high = (((self.high - self.half_range) << 1) + 1) & self.mask
                self.value = ((self.value - self.half_range) << 1 | self._read_next_bit()) & self.mask
                log.debug(f"      -> V={self.value:X}")

            # E3
            elif self.low >= self.quarter_range and self.high < self.three_q_range:
                log.debug(f"    SCALE E3: L={self.low:X}, H={self.high:X}")
                self.low = ((self.low - self.quarter_range) << 1) & self.mask
                self.high = (((self.high - self.quarter_range) << 1) + 1) & self.mask
                self.value = ((self.value - self.quarter_range) << 1 | self._read_next_bit()) & self.mask
                log.debug(f"      -> V={self.value:X}")
            
            else:
                break
                
    def _read_next_bit(self) -> int:
        bit = self.reader.read_bit()
        bit_to_return = 0 if bit == -1 else bit
        log.debug(f"    BIT: Read {bit_to_return} (raw={bit})")
        return bit_to_return

    def decode_symbol(self) -> int:
        log.debug(f"--- DECODE step ---")
        current_range = self.high - self.low + 1
        total = self.model.get_total_count()
        
        log.debug(f"  State IN:  L={self.low:X}, H={self.high:X}, R={current_range:X}")
        log.debug(f"  Value:     V={self.value:X}")

        scaled_value = ((self.value - self.low) * total) // current_range
        log.debug(f"  ScaledVal: {scaled_value} (total={total})")
        
        symbol, (sym_low, sym_high, total_check) = self.model.get_symbol_from_value(scaled_value)
        char_repr = f"'{chr(symbol)}'" if 32 <= symbol <= 126 else f"[{symbol}]"
        if symbol == self.model.get_eof_symbol(): char_repr = "[EOF]"
        log.debug(f"  Found symbol={char_repr}")
        log.debug(f"  Model range: [{sym_low}, {sym_high}) / {total_check}")

        new_low = self.low + (current_range * sym_low // total)
        new_high = self.low + (current_range * sym_high // total) - 1
        log.debug(f"  State NEW: L={new_low:X}, H={new_high:X}")
        
        self.low = new_low
        self.high = new_high
        
        self._scale()
        log.debug(f"  State OUT: L={self.low:X}, H={self.high:X}, V={self.value:X}")
        
        self.model.update_model(symbol)
        
        return symbol