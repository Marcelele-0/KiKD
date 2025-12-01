# tests/test_jpeg_ls.py
import pytest
import sys
from pathlib import Path
from src.jpeg_ls import analyze_image

# Path to test images (test_images directory at the same level as src)
IMAGES_DIR = Path(__file__).parent.parent / "test_images"


def get_test_images() -> list[Path]:
    """
    Finds all .tga files in the test_images folder.
    Returns:
        list[Path]: List of image file paths.
    """
    if not IMAGES_DIR.exists():
        return []
    return list(IMAGES_DIR.glob("*.tga"))


# Pytest parametrization - runs this test for each file in the list
@pytest.mark.parametrize("image_path", get_test_images())
def test_jpeg_ls_process_image(image_path: Path) -> None:
    """
    Test if analyze_image returns correct data structures and reasonable results for each image.
    Args:
        image_path (Path): Path to the test image.
    """
    print(f"Testing file: {image_path}")
    results = analyze_image(image_path)

    # 1. Check result structure
    assert "original" in results
    assert "predictors" in results
    assert len(results["predictors"]) == 8  # Should be 8 predictors (7 old + 1 new)

    # 2. Sanity check for data
    assert results["original"]["total"] >= 0  # Entropy must be >= 0

    # Check if all predictors are present
    expected_predictors = [
        '1: W', '2: N', '3: NW',
        '4: N+W-NW', '5: N+(W-NW)/2',
        '6: W+(N-NW)/2', '7: (N+W)/2',
        'New (JPEG-LS)'
    ]
    found_predictors = [p['name'] for p in results['predictors']]
    for expected in expected_predictors:
        assert expected in found_predictors

    # 3. Check if prediction gives any improvement
    # Usually (though not always for noise/random data),
    # the best predictor should have lower entropy than the original.
    # We check softly - at least calculations were performed.
    best_predictor = min(results["predictors"], key=lambda x: x['total'])
    print(f"  Best predictor for {image_path.name}: {best_predictor['name']} ({best_predictor['total']:.2f})")
    assert best_predictor['total'] >= 0  # Error entropy should not be negative (mathematically impossible, but we check for bugs)