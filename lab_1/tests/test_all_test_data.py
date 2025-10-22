"""Integration test that analyzes all test data files and displays results in a table."""
import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from FileAnalyser import FileAnalyzer


class TestAllTestDataFiles:
    """Integration tests for all files in test_data directory."""

    @pytest.fixture
    def test_data_dir(self):
        """Get the test_data directory path."""
        return Path(__file__).parent.parent / "test_data"

    def test_analyze_all_files_with_table(self, test_data_dir, capsys):
        """Analyze all test data files and display results in a formatted table."""
        # Get all files in test_data directory
        test_files = sorted(test_data_dir.glob("*"))
        test_files = [f for f in test_files if f.is_file()]

        if not test_files:
            pytest.skip("No test data files found")

        results = []

        # Analyze each file
        for file_path in test_files:
            analyzer = FileAnalyzer(str(file_path))
            analyzer.read_and_count_data()

            entropy = analyzer.calculate_entropy()
            cond_entropy = analyzer.calculate_conditional_entropy()
            reduction = entropy - cond_entropy
            file_size = file_path.stat().st_size

            results.append({
                'filename': file_path.name,
                'size': file_size,
                'entropy': entropy,
                'cond_entropy': cond_entropy,
                'reduction': reduction
            })

        # Print results in a nice table format
        print("\n" + "="*100)
        print("ENTROPY ANALYSIS - ALL TEST FILES")
        print("="*100)
        
        # Header
        print(f"\n{'Filename':<45} {'Size (bytes)':<15} {'H(Y)':<12} {'H(Y|X)':<12} {'Reduction':<12}")
        print("-"*100)

        # Data rows
        for result in results:
            print(f"{result['filename']:<45} "
                  f"{result['size']:<15,} "
                  f"{result['entropy']:<12.4f} "
                  f"{result['cond_entropy']:<12.4f} "
                  f"{result['reduction']:<12.4f}")

        print("="*100)
        
        # Analysis summary
        print("\nINTERPRETATION:")
        print("-"*100)
        
        for result in results:
            filename = result['filename']
            h_y = result['entropy']
            h_y_x = result['cond_entropy']
            reduction = result['reduction']
            
            print(f"\n📁 {filename}:")
            
            # Analyze entropy
            if h_y < 0.1:
                print(f"   • H(Y) ≈ 0.00: File contains only one unique byte value (constant data)")
            elif h_y > 7.9:
                print(f"   • H(Y) ≈ 8.00: Maximum entropy - all 256 byte values appear with equal frequency")
            elif h_y > 6.0:
                print(f"   • H(Y) = {h_y:.2f}: High entropy - diverse byte distribution")
            elif h_y > 3.0:
                print(f"   • H(Y) = {h_y:.2f}: Moderate entropy - typical for text or structured data")
            else:
                print(f"   • H(Y) = {h_y:.2f}: Low entropy - limited symbol set or repetitive data")
            
            # Analyze conditional entropy
            if h_y_x < 0.1:
                print(f"   • H(Y|X) ≈ 0.00: Perfect predictability - next byte is deterministic given previous")
            elif h_y_x > 7.9:
                print(f"   • H(Y|X) ≈ 8.00: No predictability - random/independent data")
            elif reduction > 1.0:
                print(f"   • H(Y|X) = {h_y_x:.2f}: Good predictability - strong sequential dependencies")
            else:
                print(f"   • H(Y|X) = {h_y_x:.2f}: Weak predictability - limited sequential patterns")
            
            # Analyze reduction
            if reduction < 0.1:
                print(f"   • Reduction ≈ 0.00: Knowing previous byte provides no information (random data)")
                print(f"   • 💾 Compressibility: VERY LOW - nearly impossible to compress")
            elif reduction > 5.0:
                print(f"   • Reduction = {reduction:.2f}: Huge information gain from context")
                print(f"   • 💾 Compressibility: EXCELLENT - differential/pattern encoding highly effective")
            elif reduction > 1.0:
                print(f"   • Reduction = {reduction:.2f}: Significant information gain from context")
                print(f"   • 💾 Compressibility: GOOD - statistical methods work well")
            else:
                print(f"   • Reduction = {reduction:.2f}: Limited information gain from context")
                print(f"   • 💾 Compressibility: LOW - minimal structure to exploit")
            
            # Identify file type
            if h_y < 0.1 and h_y_x < 0.1:
                print(f"   🔍 Type: Constant data (e.g., all zeros/ones)")
            elif h_y > 7.9 and h_y_x < 0.5:
                print(f"   🔍 Type: Sequential pattern (e.g., 00 01 02 ... FF repeated)")
            elif h_y > 7.9 and h_y_x > 7.9:
                print(f"   🔍 Type: Random/cryptographic data (e.g., /dev/urandom, encrypted file)")
            elif 4.0 < h_y < 6.0 and reduction > 1.0:
                print(f"   🔍 Type: Natural language text (structural patterns in language)")
            else:
                print(f"   🔍 Type: Mixed/structured data")

        print("\n" + "="*100 + "\n")

        # Assertions to ensure test passes if files were analyzed
        assert len(results) > 0, "Should have analyzed at least one file"
        for result in results:
            assert result['entropy'] >= 0, f"Entropy should be non-negative for {result['filename']}"
            assert result['cond_entropy'] >= 0, f"Conditional entropy should be non-negative for {result['filename']}"
            assert result['entropy'] >= result['cond_entropy'] - 0.001, \
                f"H(Y) should be >= H(Y|X) for {result['filename']}"

    def test_specific_files_expected_values(self, test_data_dir):
        """Test that specific test files have expected entropy characteristics."""
        test_files = {
            'test1.bin': {'entropy_range': (0.0, 0.1), 'cond_entropy_range': (0.0, 0.1)},
            'test2.bin': {'entropy_range': (7.9, 8.0), 'cond_entropy_range': (0.0, 0.1)},
            'test3.bin': {'entropy_range': (7.9, 8.0), 'cond_entropy_range': (7.9, 8.0)},
            'pan.txt': {'entropy_range': (4.0, 6.0), 'cond_entropy_range': (2.0, 5.0)},
        }

        for filename, expected in test_files.items():
            file_path = test_data_dir / filename
            
            if not file_path.exists():
                pytest.skip(f"Test file {filename} not found")
                continue

            analyzer = FileAnalyzer(str(file_path))
            analyzer.read_and_count_data()

            entropy = analyzer.calculate_entropy()
            cond_entropy = analyzer.calculate_conditional_entropy()

            # Check entropy is in expected range
            assert expected['entropy_range'][0] <= entropy <= expected['entropy_range'][1], \
                f"{filename}: Entropy {entropy:.4f} not in expected range {expected['entropy_range']}"

            # Check conditional entropy is in expected range
            assert expected['cond_entropy_range'][0] <= cond_entropy <= expected['cond_entropy_range'][1], \
                f"{filename}: Conditional entropy {cond_entropy:.4f} not in expected range {expected['cond_entropy_range']}"

    def test_entropy_inequality_holds(self, test_data_dir):
        """Test that H(Y|X) <= H(Y) holds for all files (fundamental property)."""
        test_files = [f for f in test_data_dir.glob("*") if f.is_file()]

        for file_path in test_files:
            analyzer = FileAnalyzer(str(file_path))
            analyzer.read_and_count_data()

            entropy = analyzer.calculate_entropy()
            cond_entropy = analyzer.calculate_conditional_entropy()

            # Allow small floating point tolerance
            assert cond_entropy <= entropy + 0.001, \
                f"{file_path.name}: H(Y|X)={cond_entropy:.4f} > H(Y)={entropy:.4f} violates information theory"

    def test_compare_files_characteristics(self, test_data_dir):
        """Compare characteristics between different test files."""
        files = ['test1.bin', 'test2.bin', 'test3.bin']
        analyses = {}

        for filename in files:
            file_path = test_data_dir / filename
            if not file_path.exists():
                continue

            analyzer = FileAnalyzer(str(file_path))
            analyzer.read_and_count_data()

            analyses[filename] = {
                'entropy': analyzer.calculate_entropy(),
                'cond_entropy': analyzer.calculate_conditional_entropy(),
            }

        if len(analyses) < 3:
            pytest.skip("Not all test files available")

        # test1.bin should have lowest entropy
        if 'test1.bin' in analyses:
            assert analyses['test1.bin']['entropy'] < 1.0, \
                "test1.bin should have very low entropy (constant data)"

        # test2.bin and test3.bin should have high entropy
        if 'test2.bin' in analyses:
            assert analyses['test2.bin']['entropy'] > 7.9, \
                "test2.bin should have maximum entropy"

        if 'test3.bin' in analyses:
            assert analyses['test3.bin']['entropy'] > 7.9, \
                "test3.bin should have maximum entropy"

        # test2.bin should have much lower conditional entropy than test3.bin
        if 'test2.bin' in analyses and 'test3.bin' in analyses:
            assert analyses['test2.bin']['cond_entropy'] < analyses['test3.bin']['cond_entropy'] - 5.0, \
                "test2.bin (pattern) should have much lower H(Y|X) than test3.bin (random)"
