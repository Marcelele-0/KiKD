
import math
from collections import Counter

def calculate_entropy(data_list: list) -> float:
    """
    Oblicza entropię (H) dla listy symboli (mogą to być bajty lub indeksy).
    """
    if not data_list:
        return 0.0
        
    counts = Counter(data_list)
    total_symbols = len(data_list)
    entropy = 0.0
    
    for count in counts.values():
        probability = count / total_symbols
        entropy -= probability * math.log2(probability)
        
    return entropy