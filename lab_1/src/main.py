import os
import sys

from FileAnalyser import FileAnalyzer


def main():
    """Main function to handle command-line arguments and run the analysis.
    """
    if len(sys.argv) != 2:
        print("Usage: python analyzer.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"Error: The provided path '{file_path}' is not a valid file.")
        sys.exit(1)

    analyzer = FileAnalyzer(file_path)
    analyzer.read_and_count_data()

    entropy = analyzer.calculate_entropy()
    cond_entropy = analyzer.calculate_conditional_entropy()

    analyzer.print_results(entropy, cond_entropy)

if __name__ == "__main__":
    main()
