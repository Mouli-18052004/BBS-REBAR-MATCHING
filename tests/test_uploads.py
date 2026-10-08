"""Upload decoding checks, independent of candidate IDs."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageOps

import matcher as vm


class ImageUploadTests(unittest.TestCase):
    def test_alpha_is_composited_not_discarded(self):
        rgba = np.zeros((50, 50, 4), dtype=np.uint8)
        rgba[8:42, 20:24, 3] = 255
        rgba[10, 24, 3] = 128
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'drawing.png'
            Image.fromarray(rgba).save(path)
            actual = vm.read_customer_image(path)
        self.assertEqual(actual[0, 0].tolist(), [255]*3)
        self.assertEqual(actual[20, 21].tolist(), [0]*3)
        self.assertEqual(actual[10, 24].tolist(), [127]*3)

    def test_palette_transparency(self):
        im = Image.new('P', (20, 20), 0)
        im.putpalette([0, 0, 0, 0, 0, 0] + [255, 255, 255] * 254)
        im.putpixel((10, 10), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'palette.png'
            im.save(path, transparency=0)
            actual = vm.read_customer_image(path)
        self.assertEqual(actual[0, 0].tolist(), [255]*3)
        self.assertEqual(actual[10, 10].tolist(), [0]*3)

    def test_exif_orientation_and_unicode_path(self):
        im = Image.new('RGB', (40, 20), 'white')
        for x in range(10):
            for y in range(20):
                im.putpixel((x, y), (0, 0, 0))
        exif = im.getexif()
        exif[274] = 6
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '\u0930\u0947\u0916\u093e.png'
            im.save(path, exif=exif)
            actual = vm.read_customer_image(path)
        self.assertEqual(actual.shape, (40, 20, 3))
        self.assertTrue(np.all(actual[:10] == 0))

    def test_sixteen_bit_grayscale_keeps_dynamic_range(self):
        pixels = np.array([[0, 32768, 65535]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'depth.png'
            Image.fromarray(pixels).save(path)
            actual = vm.read_customer_image(path)
        self.assertEqual(actual[0, :, 0].tolist(), [0, 128, 255])

    def test_synthetic_demo_pixels_do_not_change(self):
        for path in (vm.PROJECT_ROOT / 'demo_data/geometry').glob('*.png'):
            with self.subTest(path=path.name):
                np.testing.assert_array_equal(vm.read_customer_image(path), cv2.imread(str(path)))

    def test_invalid_and_missing_files_report_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'invalid.png'
            with self.assertRaises(FileNotFoundError):
                vm.read_customer_image(path)
            path.write_bytes(b'not an image')
            with self.assertRaisesRegex(ValueError, 'Cannot decode'):
                vm.read_customer_image(path)


if __name__ == '__main__':
    unittest.main()
