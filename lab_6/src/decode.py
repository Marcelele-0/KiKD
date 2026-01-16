import sys
import pickle
import numpy as np
from PIL import Image
from src.transform import inverse_haar_transform, dpcm_decode
from src.lbg_1d import dequantize_1d

def main():
    if len(sys.argv) != 3:
        print("Użycie: python decode.py <encoded.bin> <output.tga>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Dekodowanie: {input_path}...")
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    width, height = data['width'], data['height']
    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for ch in range(3):
        ch_data = data['channels'][ch]
        
        # 1. Dekwantyzacja (Indeksy -> Wartości przybliżone)
        low_diff = dequantize_1d(ch_data['inds_low'], ch_data['cb_low'])
        high = dequantize_1d(ch_data['inds_high'], ch_data['cb_high'])
        
        # 2. Odwrócenie DPCM (tylko dla pasma Low)
        low = dpcm_decode(low_diff)
        
        # 3. Odwrócenie transformaty (L, H -> Piksele)
        reconstructed_ch = inverse_haar_transform(low, high, (height, width))
        
        final_image[:, :, ch] = reconstructed_ch
        
    img = Image.fromarray(final_image)
    img.save(output_path)
    print(f"Zapisano odkodowany obraz: {output_path}")

if __name__ == "__main__":
    main()