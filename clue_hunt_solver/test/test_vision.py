from pathlib import Path
import unittest

import cv2

from clue_hunt_solver.vision import decode_qr_codes, detect_markers, pair_marker_with_qr


FIXTURES = Path(__file__).resolve().parent / 'fixtures'


class VisionTests(unittest.TestCase):
    def test_every_practice_texture(self):
        expected = {
            'board_practice_b1.png': (1, 'HUNT:1:7196:'),
            'board_practice_b2.png': (2, 'HUNT:2:7F49:'),
            'board_practice_b3.png': (3, 'HUNT:3:BCB8:'),
            'board_practice_b4.png': (4, 'HUNT:4:EC9E:'),
            'board_practice_b5.png': (5, 'HUNT:5:1756:'),
            'board_practice_d7.png': (7, 'HUNT:7:70DD:'),
            'board_practice_x4.png': (4, 'HUNT:4:07A8:'),
        }
        textures = FIXTURES.glob('*.png')
        seen = 0
        for path in textures:
            image = cv2.imread(str(path))
            markers = detect_markers(image, 0.24)
            qr_codes = decode_qr_codes(image)
            marker_id, prefix = expected[path.name]
            self.assertIn(marker_id, [marker.marker_id for marker in markers])
            self.assertTrue(any(qr.text.startswith(prefix) for qr in qr_codes))
            marker = next(item for item in markers if item.marker_id == marker_id)
            self.assertIsNotNone(pair_marker_with_qr(marker, qr_codes))
            seen += 1
        self.assertEqual(seen, len(expected))


if __name__ == '__main__':
    unittest.main()
