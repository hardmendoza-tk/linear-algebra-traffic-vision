import unittest
import numpy as np
import cv2
from src.detector import detectar_componentes, Parametros


class ConteoTests(unittest.TestCase):
    def setUp(self):
        self.roi = np.ones((200, 300), np.uint8)
        self.p = Parametros(area_minima=100, kernel_cierre=11)

    def test_un_auto_grande_no_se_multiplica(self):
        img = np.zeros_like(self.roi)
        cv2.rectangle(img, (35, 30), (110, 115), 1, -1)
        boxes, _, _ = detectar_componentes(img, self.roi, self.p)
        self.assertEqual(len(boxes), 1)

    def test_dos_autos_separados(self):
        img = np.zeros_like(self.roi)
        cv2.rectangle(img, (35, 30), (95, 75), 1, -1)
        cv2.rectangle(img, (160, 100), (220, 150), 1, -1)
        boxes, _, _ = detectar_componentes(img, self.roi, self.p)
        self.assertEqual(len(boxes), 2)

    def test_fragmentos_cercanos_de_un_auto(self):
        img = np.zeros_like(self.roi)
        cv2.rectangle(img, (35, 30), (95, 60), 1, -1)
        cv2.rectangle(img, (35, 66), (95, 95), 1, -1)
        boxes, _, _ = detectar_componentes(img, self.roi, self.p)
        self.assertEqual(len(boxes), 1)

    def test_ruido_no_cuenta(self):
        img = np.zeros_like(self.roi)
        cv2.rectangle(img, (30, 30), (35, 35), 1, -1)
        boxes, _, _ = detectar_componentes(img, self.roi, self.p)
        self.assertEqual(len(boxes), 0)

    def test_fuera_de_roi_no_cuenta(self):
        img = np.zeros_like(self.roi)
        cv2.rectangle(img, (170, 50), (250, 125), 1, -1)
        roi = np.zeros_like(self.roi)
        roi[:, :150] = 1
        boxes, _, _ = detectar_componentes(img, roi, self.p)
        self.assertEqual(len(boxes), 0)

if __name__ == '__main__':
    unittest.main()
