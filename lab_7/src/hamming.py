# Generator Matrix G for Hamming (7,4) [I | P]
G = [
    [1, 1, 0, 1], # P1
    [1, 0, 1, 1], # P2
    [1, 0, 0, 0], # D1
    [0, 1, 1, 1], # P3
    [0, 1, 0, 0], # D2
    [0, 0, 1, 0], # D3
    [0, 0, 0, 1], # D4
]

# Parity Check Matrix H
H = [
    [1, 0, 1, 0, 1, 0, 1], # 1
    [0, 1, 1, 0, 0, 1, 1], # 2
    [0, 0, 0, 1, 1, 1, 1], # 4
]

class HammingCodec:
    def __init__(self):
        self.enc_table = self._build_encoding_table()
        self.dec_table = self._build_decoding_table()

    def _build_encoding_table(self) -> list[int]:
        """Maps 4-bit data (0-15) to 8-bit codeword (SECDED)."""
        table = []
        for nibble in range(16):
            d = [(nibble >> i) & 1 for i in range(3, -1, -1)]
            
            # Calculate 7 Hamming bits
            code_bits = []
            for row in G:
                bit_sum = sum(r * val for r, val in zip(row, d))
                code_bits.append(bit_sum % 2)
            
            # Add 8th parity bit
            p8 = sum(code_bits) % 2
            code_bits.append(p8)
            
            # Convert to byte
            encoded_byte = 0
            for bit in code_bits:
                encoded_byte = (encoded_byte << 1) | bit
            table.append(encoded_byte)
        return table

    def _build_decoding_table(self) -> list[tuple[int, int]]:
        """
        Returns: (decoded_nibble, status)
        Status: 0=OK, 1=Fixed 1 error, 2=Detected 2 errors
        """
        table = []
        for byte_val in range(256):
            bits = [(byte_val >> i) & 1 for i in range(7, -1, -1)]
            cw_7 = bits[:7]
            p8_rec = bits[7]
            
            # 1. Syndrome
            syndrome = 0
            for i, row in enumerate(H):
                s_bit = sum(h * b for h, b in zip(row, cw_7)) % 2
                syndrome |= (s_bit << i)
            
            # 2. Overall parity
            p8_calc = sum(cw_7) % 2
            p_error = (p8_calc != p8_rec)

            status = 0
            
            if syndrome == 0:
                if p_error:
                    status = 1 # Error on parity bit only
                else:
                    status = 0 # OK
            else:
                if p_error:
                    # Single error -> Fix it
                    error_pos = syndrome - 1
                    cw_7[error_pos] ^= 1
                    status = 1
                else:
                    # Double error
                    status = 2 
            
            # Extract data bits
            decoded_nibble = 0
            data_indices = [2, 4, 5, 6]
            for idx in data_indices:
                decoded_nibble = (decoded_nibble << 1) | cw_7[idx]
            
            table.append((decoded_nibble, status))
        return table

# Global instance
_codec = HammingCodec()

def encode_nibble(nibble: int) -> int:
    return _codec.enc_table[nibble & 0x0F]

def decode_byte(byte_val: int) -> tuple[int, int]:
    return _codec.dec_table[byte_val & 0xFF]