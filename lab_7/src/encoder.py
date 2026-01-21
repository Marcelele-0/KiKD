import sys
from src.hamming import encode_nibble

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.encoder <input> <output>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            while True:
                byte = f_in.read(1)
                if not byte:
                    break
                
                val = byte[0]
                high = (val >> 4) & 0x0F
                low = val & 0x0F
                
                f_out.write(bytes([encode_nibble(high), encode_nibble(low)]))
                
        print(f"Encoded: {input_path} -> {output_path}")
        
    except FileNotFoundError:
        print("Error: File not found.")

if __name__ == "__main__":
    main()