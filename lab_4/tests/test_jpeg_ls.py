import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path to import jpeg_ls
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import jpeg_ls

class TestJpegLs(unittest.TestCase):

    # Entropy tests removed as per user request

    def test_get_predictors_keys(self):
        """Check if all 8 predictors are present."""
        W = np.zeros((3, 3))
        N = np.zeros((3, 3))
        NW = np.zeros((3, 3))
        preds = jpeg_ls.get_predictors(W, N, NW)
        expected_keys = [
            '1: W', '2: N', '3: NW', '4: N+W-NW',
            '5: N+(W-NW)/2', '6: W+(N-NW)/2', '7: (N+W)/2',
            'New (JPEG-LS)'
        ]
        for key in expected_keys:
            self.assertIn(key, preds)

    def test_get_predictors_values(self):
        """Check calculation logic for simple predictors."""
        # Setup simple 1x1 arrays
        W = np.array([[10]])
        N = np.array([[20]])
        NW = np.array([[5]])
        
        preds = jpeg_ls.get_predictors(W, N, NW)
        
        self.assertEqual(preds['1: W'][0,0], 10)
        self.assertEqual(preds['2: N'][0,0], 20)
        self.assertEqual(preds['3: NW'][0,0], 5)
        self.assertEqual(preds['4: N+W-NW'][0,0], 20 + 10 - 5) # 25
        self.assertEqual(preds['7: (N+W)/2'][0,0], (20 + 10) // 2) # 15

    def test_predictor_new_jpeg_ls(self):
        """Test the New (JPEG-LS) predictor logic."""
        # Case 1: NW >= max(W, N) -> min(W, N)
        # NW=20, W=10, N=15. Max(10,15)=15. 20 >= 15 -> True. Result min(10,15)=10
        W = np.array([[10]])
        N = np.array([[15]])
        NW = np.array([[20]])
        preds = jpeg_ls.get_predictors(W, N, NW)
        self.assertEqual(preds['New (JPEG-LS)'][0,0], 10)

        # Case 2: NW <= min(W, N) -> max(W, N)
        # NW=5, W=10, N=15. Min(10,15)=10. 5 <= 10 -> True. Result max(10,15)=15
        W = np.array([[10]])
        N = np.array([[15]])
        NW = np.array([[5]])
        preds = jpeg_ls.get_predictors(W, N, NW)
        self.assertEqual(preds['New (JPEG-LS)'][0,0], 15)

        # Case 3: Otherwise -> W + N - NW
        # NW=12, W=10, N=15. Min=10, Max=15. 12 is between. Result 10+15-12 = 13
        W = np.array([[10]])
        N = np.array([[15]])
        NW = np.array([[12]])
        preds = jpeg_ls.get_predictors(W, N, NW)
        self.assertEqual(preds['New (JPEG-LS)'][0,0], 13)

    @patch('PIL.Image.open')
    def test_analyze_image_structure(self, mock_open):
        """Test analyze_image returns correct structure with mocked image."""
        # Create a fake 4x4 image
        fake_img = MagicMock()
        fake_img.convert.return_value = fake_img
        # 4x4 array with random values
        fake_arr = np.array([
            [[10, 10, 10], [20, 20, 20]],
            [[30, 30, 30], [40, 40, 40]]
        ], dtype=np.int32)
        
        # When np.array(img) is called, we can't easily mock it if it takes the mock object directly.
        # However, PIL.Image.open returns an object that is then converted to array.
        # A common way to mock this integration is to mock np.array or ensure the mock object behaves like an array interface,
        # but simpler is to mock the whole logic or just ensure Image.open returns something that np.array accepts.
        # Actually, np.array(mock_obj) might fail.
        # Let's patch numpy.array inside jpeg_ls if possible, OR better:
        # Since analyze_image calls `img = Image.open(...)` then `arr = np.array(img, ...)`
        # We can make `img` behave enough like an image.
        # But `np.array(mock)` returns an array of the mock, not what we want.
        
        # Strategy: We can't easily mock np.array call inside the function without patching np.array, which is risky.
        # Alternative: The code does `img = Image.open(input_file).convert('RGB')` then `np.array(img)`.
        # If we make `Image.open` return a real (small) PIL Image, it will work perfectly.
        
        from PIL import Image as RealImage
        # Create a real 2x2 image
        real_small_img = RealImage.new('RGB', (2, 2), color='red')
        mock_open.return_value = real_small_img
        
        results = jpeg_ls.analyze_image("dummy_path.tga")
        
        self.assertIn("original", results)
        self.assertIn("predictors", results)
        self.assertIn("total", results["original"])
        self.assertEqual(len(results["predictors"]), 8)
        self.assertEqual(results["predictors"][0]["name"], "1: W")

if __name__ == '__main__':
    unittest.main()