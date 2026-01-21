import sys
import numpy as np
from PIL import Image

def calculate_mse(orig: np.ndarray, recon: np.ndarray) -> float:
    return float(np.mean((orig.astype(float) - recon.astype(float)) ** 2))

def calculate_snr(orig: np.ndarray, mse: float) -> float:
    if mse == 0:
        return float('inf')
    signal_power = np.mean(orig.astype(float) ** 2)
    return 10 * np.log10(signal_power / mse)

def main():
    if len(sys.argv) != 3:
        print("Użycie: python metrics.py <oryginal.tga> <odkodowany.tga>")
        sys.exit(1)
        
    path_orig = sys.argv[1]
    path_recon = sys.argv[2]
    
    img_orig = np.array(Image.open(path_orig).convert('RGB'))
    img_recon = np.array(Image.open(path_recon).convert('RGB'))
    
    if img_orig.shape != img_recon.shape:
        print("UWAGA: Obrazy mają różne wymiary!")
        # Przycinamy do mniejszego (na wypadek problemów z paddingiem)
        h = min(img_orig.shape[0], img_recon.shape[0])
        w = min(img_orig.shape[1], img_recon.shape[1])
        img_orig = img_orig[:h, :w, :]
        img_recon = img_recon[:h, :w, :]

    print("\n--- Analiza Błędu ---")
    
    # Cały obraz
    mse_total = calculate_mse(img_orig, img_recon)
    snr_total = calculate_snr(img_orig, mse_total)
    print(f"Cały obraz -> MSE: {mse_total:.4f}, SNR: {snr_total:.4f} dB")
    
    print("-" * 40)
    
    # Kanały
    channels = ['R (Czerwony)', 'G (Zielony) ', 'B (Niebieski)']
    for i in range(3):
        mse_c = calculate_mse(img_orig[:,:,i], img_recon[:,:,i])
        snr_c = calculate_snr(img_orig[:,:,i], mse_c)
        print(f"{channels[i]} -> MSE: {mse_c:.4f}, SNR: {snr_c:.4f} dB")

if __name__ == "__main__":
    main()