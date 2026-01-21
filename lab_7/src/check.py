import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.check <file1> <file2>")
        sys.exit(1)

    path1 = sys.argv[1]
    path2 = sys.argv[2]
    diff_blocks = 0

    try:
        with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
            while True:
                b1 = f1.read(1)
                b2 = f2.read(1)
                if not b1 or not b2: break
                
                v1, v2 = b1[0], b2[0]
                if (v1 >> 4) != (v2 >> 4): diff_blocks += 1
                if (v1 & 0xF) != (v2 & 0xF): diff_blocks += 1
                
        print(f"Compare: {path1} vs {path2}")
        print(f"Different 4-bit blocks: {diff_blocks}")
        
    except FileNotFoundError:
        print("Error: File not found.")

if __name__ == "__main__":
    main()