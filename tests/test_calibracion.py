import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from src.calibracion import mascara_poligono, guardar_calibracion
from src.detector import cargar_calibracion, detectar_componentes, Parametros


class CalibracionYForma(unittest.TestCase):
    def setUp(self):
        self.imagen = np.full((200, 300, 3), 75, dtype=np.uint8)
        self.a = [(10, 10), (140, 10), (140, 180), (10, 180)]
        self.b = [(160, 10), (290, 10), (290, 180), (160, 180)]

    def test_calibracion_guardar_y_cargar(self):
        with tempfile.TemporaryDirectory() as tmp:
            guardar_calibracion(Path(tmp), self.imagen, self.a, self.b)
            fondo, H, roi = cargar_calibracion(Path(tmp))
            self.assertEqual(fondo.shape, (200, 300))
            self.assertTrue(np.array_equal(H, np.eye(3)))
            self.assertEqual(np.count_nonzero(roi['A'] & roi['B']), 0)

    def test_poligono_demasiado_corto(self):
        with self.assertRaises(ValueError):
            mascara_poligono((200,300), [(1,1),(2,2)])

    def test_no_permitir_rois_solapadas(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                guardar_calibracion(tmp, self.imagen, self.a, self.a)

    def test_objeto_compacto_cuenta_uno(self):
        img = np.zeros((200,300), np.uint8)
        cv2.rectangle(img, (30,40), (100,90), 1, -1)
        found, _, _ = detectar_componentes(img, np.ones_like(img), Parametros())
        self.assertEqual(len(found),1)

    def test_forma_estrecha_se_descarta(self):
        img = np.zeros((200,300), np.uint8)
        cv2.rectangle(img, (20,20), (27,170), 1, -1)
        found, _, _ = detectar_componentes(img, np.ones_like(img), Parametros())
        self.assertEqual(len(found),0)

    def test_forma_irregular_ocupacion_baja(self):
        img = np.zeros((200,300), np.uint8)
        cv2.rectangle(img,(25,30),(35,150),1,-1)
        cv2.rectangle(img,(25,140),(140,150),1,-1)
        p=Parametros(ocupacion_min=.6,kernel_cierre=1,ancho_minimo=5,alto_minimo=5)
        found, _, _=detectar_componentes(img,np.ones_like(img),p)
        self.assertEqual(len(found),0)

if __name__ == '__main__':
    unittest.main()
