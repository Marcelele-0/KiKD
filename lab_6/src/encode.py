import sys
import pickle
import numpy as np
from PIL import Image
from src.transform import apply_haar_transform, dpcm_encode
from src.lbg_1d import train_lbg_1d, quantize_1d

def main():
    if len(sys.argv) != 4:
        print("Użycie: python encode.py <input.tga> <output.bin> <k_bits>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        k_bits = int(sys.argv[3])
        if not (1 <= k_bits <= 7):
            raise ValueError("K musi być w zakresie 1-7")
    except ValueError as e:
        print(f"Błąd argumentu: {e}")
        sys.exit(1)

    print(f"Kodowanie: {input_path} (K={k_bits})...")
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    height, width, _ = arr.shape

    # Struktura pliku wynikowego
    encoded_data = {
        'width': width,
        'height': height,
        'k': k_bits,
        'channels': []
    }

    # Przetwarzamy każdy kanał R, G, B osobno
    for ch in range(3):
        channel_pixels = arr[:, :, ch]
        
        # 1. Transformata (podział na L i H)
        low, high = apply_haar_transform(channel_pixels)
        
        # 2. DPCM tylko dla pasma Low (wymaganie na ocenę 4/5)
        low_diff = dpcm_encode(low)
        
        # Pasmo High zostawiamy wprost (jest już "szumem", nie ma korelacji sąsiadów)
        
        # 3. Trening kwantyzatorów (LBG 1D) - WYMAGANIE NA 5.0
        # Uczymy się, jakie wartości dominują w różnicach i szumie
        # Używamy co 10-tej próbki dla przyspieszenia treningu
        print(f"  Kanał {ch}: Trenowanie codebooków...")
        cb_low = train_lbg_1d(low_diff[::10], k_bits)
        cb_high = train_lbg_1d(high[::10], k_bits)
        
        # 4. Kwantyzacja (zamiana wartości na indeksy)
        inds_low = quantize_1d(low_diff, cb_low).astype(np.uint8)
        inds_high = quantize_1d(high, cb_high).astype(np.uint8)
        
        encoded_data['channels'].append({
            'cb_low': cb_low,     # Musimy zapisać paletę, żeby móc odkodować!
            'cb_high': cb_high,
            'inds_low': inds_low,
            'inds_high': inds_high
        })

    # Zapis do pliku binarnego
    with open(output_path, 'wb') as f:
        pickle.dump(encoded_data, f)
        
    print(f"Zakończono. Wynik zapisano w: {output_path}")

if __name__ == "__main__":
    main()