import os
import sys

import numpy as np


class FileAnalyzer:
    """A class to analyze a file and calculate its entropy and conditional entropy.
    """

    def __init__(self, file_path: str):
        """Initializes the FileAnalyzer with a given file path.

        Args:
            file_path: The path to the file to be analyzed.
        """
        self.file_path: str = file_path
        self.symbol_counts: np.ndarray = np.zeros(256, dtype=np.int64)
        self.conditional_counts: np.ndarray = np.zeros((256, 256), dtype=np.int64)

    def read_and_count_data(self) -> None:
        """Reads the file byte by byte and counts symbol frequencies
        and conditional frequencies.
        The first symbol is treated as if it were preceded by a byte with value 0.
        """
        try:
            with open(self.file_path, 'rb') as f:
                data = f.read()
                if not data:
                    print("Error: The file is empty.")
                    return

                # Assuming the first character is preceded by a byte of value 0
                prev_byte = 0

                for current_byte in data:
                    self.symbol_counts[current_byte] += 1
                    self.conditional_counts[prev_byte, current_byte] += 1
                    prev_byte = current_byte

        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
            sys.exit(1)
        except IOError:
            print(f"Error: Could not read the file '{self.file_path}'.")
            sys.exit(1)

    def calculate_entropy(self) -> float:
        """Calculates the entropy of the file's symbols.

        Returns:
            The entropy value in bits.
        """
        total_symbols = np.sum(self.symbol_counts)
        if total_symbols == 0:
            return 0.0

        probabilities = self.symbol_counts / total_symbols
        # Filter out zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]

        # H(Y) = -sum(p * log2(p))
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def calculate_conditional_entropy(self) -> float:
        """Calculates the conditional entropy H(Y|X) of the file's symbols.

        Returns:
            The conditional entropy value in bits.
        """
        total_symbols = np.sum(self.symbol_counts)
        if total_symbols == 0:
            return 0.0

        # Calculate P(x)
        p_x = self.symbol_counts / total_symbols

        # Calculate H(Y|x) for each x
        h_y_given_x = np.zeros(256, dtype=float)

        for x in range(256):
            sum_x = np.sum(self.conditional_counts[x])
            if sum_x > 0:
                # Calculate P(y|x)
                p_y_given_x = self.conditional_counts[x] / sum_x
                # Filter out zero probabilities
                p_y_given_x = p_y_given_x[p_y_given_x > 0]

                # Calculate H(Y|x) = -sum(P(y|x) * log2(P(y|x)))
                h_y_given_x[x] = -np.sum(p_y_given_x * np.log2(p_y_given_x))

        # H(Y|X) = sum(P(x) * H(Y|x))
        conditional_entropy = np.sum(p_x * h_y_given_x)
        return conditional_entropy

    def print_results(self, entropy: float, cond_entropy: float) -> None:
        """Prints the calculated entropy values in a readable format.

        Args:
            entropy: The calculated entropy.
            cond_entropy: The calculated conditional entropy.
        """
        print("\n--- Entropy Analysis ---")
        print(f"File: {self.file_path}")
        print(f"File size: {os.path.getsize(self.file_path)} bytes")
        print("------------------------")
        print(f"Entropy H(Y): {entropy:.4f} bits/symbol")
        print(f"Conditional Entropy H(Y|X): {cond_entropy:.4f} bits/symbol")
        print(f"Reduction in uncertainty: {entropy - cond_entropy:.4f} bits/symbol")
        print("------------------------\n")

