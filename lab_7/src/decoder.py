import sys
from src.hamming import decode_byte

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.decoder <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    double_err = 0
    corrected_err = 0

    try:
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            while True:
                chunk = f_in.read(2)
                if len(chunk) < 2:
                    break
                
                nib_h, stat_h = decode_byte(chunk[0])
                nib_l, stat_l = decode_byte(chunk[1])
                
                if stat_h == 2: double_err += 1
                elif stat_h == 1: corrected_err += 1
                
                if stat_l == 2: double_err += 1
                elif stat_l == 1: corrected_err += 1
                
                byte_val = (nib_h << 4) | nib_l
                f_out.write(bytes([byte_val]))
        
        print(f"Decoded: {input_path} -> {output_path}")
        print(f"Double errors (detected): {double_err}")
        print(f"Single errors (corrected): {corrected_err}")

    except FileNotFoundError:
        print("Error: File not found.")

if __name__ == "__main__":
    main()