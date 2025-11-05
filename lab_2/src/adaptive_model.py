class AdaptiveModel:
    """
    Przechowuje liczniki symboli i pozwala na ich adaptacyjną aktualizację.
    Używa Drzewa Fenwicka (BIT) do szybkiego obliczania sum prefiksowych,
    co drastycznie przyspiesza operacje get_range i get_symbol_from_value.
    
    Złożoność operacji kluczowych spada z O(N) do O(log N) lub O((log N)^2).
    """
    
    MAX_TOTAL_COUNT = 16383

    def __init__(self, num_symbols=256):
        self.NUM_SYMBOLS = num_symbols
        self.EOF_SYMBOL = self.NUM_SYMBOLS
        self.NUM_CHARS = self.NUM_SYMBOLS + 1  # +1 dla EOF

        self.counts = [1] * self.NUM_CHARS
        self.total_count = self.NUM_CHARS
        self.tree = [0] * (self.NUM_CHARS + 1)
        
        for i in range(self.NUM_CHARS):
            self._tree_update(i, 1)

    def _tree_update(self, index: int, delta: int):
        """Aktualizuje drzewo BIT. Dodaje 'delta' do elementu 'index'."""
        index += 1  # Przejście na 1-indeksowanie
        while index < len(self.tree):
            self.tree[index] += delta
            index += index & (-index)  # Przejdź do rodzica

    def _tree_query(self, index: int) -> int:
        """Zwraca sumę prefiksową elementów od 0 do index."""
        if index < 0:
            return 0
        sum_val = 0
        index += 1  # Przejście na 1-indeksowanie
        while index > 0:
            sum_val += self.tree[index]
            index -= index & (-index)  # Przejdź do kolejnego węzła
        return sum_val

    def get_eof_symbol(self) -> int:
        return self.EOF_SYMBOL

    def get_total_count(self) -> int:
        return self.total_count

    def update_model(self, symbol: int):
        """Inkrementuje licznik dla symbolu i aktualizuje sumę."""
        self._tree_update(symbol, 1)
        self.counts[symbol] += 1
        self.total_count += 1
        
        if self.total_count >= self.MAX_TOTAL_COUNT:
            self.rescale_counts()

    def rescale_counts(self):
        """Przeskalowuje liczniki (wolna operacja, ale amortyzowana)."""
        self.total_count = 0
        self.tree = [0] * (self.NUM_CHARS + 1)
        
        for i in range(self.NUM_CHARS):
            self.counts[i] = max(1, self.counts[i] >> 1)
            self.total_count += self.counts[i]
            self._tree_update(i, self.counts[i])

    def get_range(self, symbol: int) -> tuple[int, int, int]:
        """Zwraca (dolny_zakres, górny_zakres, suma) dla symbolu."""
        # Suma [0...symbol-1]
        low_c = self._tree_query(symbol - 1)
        # Suma [0...symbol]
        high_c = self._tree_query(symbol)
        return low_c, high_c, self.total_count

    def get_symbol_from_value(self, scaled_value: int) -> tuple[int, (int, int, int)]:
        """Znajduje symbol odpowiadający wartości przy użyciu wyszukiwania binarnego."""
        low = 0
        high = self.NUM_SYMBOLS  # Szukamy w zakresie [0...EOF_SYMBOL]
        
        while low <= high:
            mid = (low + high) // 2
            mid_range_low = self._tree_query(mid - 1)
            mid_range_high = self._tree_query(mid)

            if mid_range_low <= scaled_value < mid_range_high:
                return mid, (mid_range_low, mid_range_high, self.total_count)
            elif scaled_value < mid_range_low:
                high = mid - 1
            else:
                low = mid + 1
                
        raise ValueError("Wartość spoza zakresu w modelu (błąd dekodera).")