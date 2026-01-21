import sys
import random

def main():
    if len(sys.argv) != 4:
        print("Usage: python -m src.noise <p> <input> <output>")
        sys.exit(1)

    try:
        p = float(sys.argv[1])
        input_path = sys.argv[2]
        output_path = sys.argv[3]
    except ValueError:
        print("Error: p must be float.")
        sys.exit(1)

    try:
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            data = bytearray(f_in.read())
            
            for i in range(len(data)):
                mask = 1
                for _ in range(8):
                    if random.random() < p:
                        data[i] ^= mask
                    mask <<= 1
                
            f_out.write(data)
            
        print(f"Noise (p={p}): {input_path} -> {output_path}")

    except FileNotFoundError:
        print("Error: File not found.")

if __name__ == "__main__":
    main()