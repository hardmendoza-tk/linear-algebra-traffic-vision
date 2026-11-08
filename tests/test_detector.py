import unittest
from pathlib import Path
import numpy as np
from src.detector import cargar_calibracion, procesar_fotograma, Parametros

BASE = Path(__file__).resolve().parents[1]

class PruebasDetector(unittest.TestCase):
    def test_calibracion(self):
        fondo, H, mascaras = cargar_calibracion(BASE/'data')
        self.assertEqual(fondo.shape, (480, 640))
        self.assertEqual(set(mascaras), {'A','B'})

    def test_frame_sin_cambios(self):
        fondo, H, mascaras = cargar_calibracion(BASE/'data')
        # H invierte la perspectiva; crear frame de origen a partir del fondo.
        H_inv = np.linalg.inv(H)
        import cv2
        frame = cv2.warpPerspective(fondo, H_inv, (640,480))
        r, _, _ = procesar_fotograma(frame, fondo, H, mascaras, Parametros())
        self.assertEqual(set(r), {'A','B'})
        for calle in r:
            self.assertGreaterEqual(r[calle]['vehiculos_aprox'], 0)
            self.assertGreaterEqual(r[calle]['ocupacion_pct'], 0)

if __name__ == '__main__':
    unittest.main()
