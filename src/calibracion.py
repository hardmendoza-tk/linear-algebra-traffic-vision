"""Calibración interactiva para cualquier escena de dos calles, sin hardware.

Se recomienda usar un fotograma SIN coches para registrar el fondo.
Las ROI se eligen con polígonos (3+ puntos) en coordenadas de la imagen.
"""
from pathlib import Path
import cv2
import numpy as np


def mascara_poligono(tamano, puntos):
    """Devuelve una máscara uint8 (0/1) a partir de al menos tres vértices."""
    alto, ancho = tamano
    pts = np.asarray(puntos, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError('Se necesitan al menos 3 puntos por calle.')
    if not (np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < ancho) and
            np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < alto)):
        raise ValueError('Hay puntos fuera de la imagen.')
    mask = np.zeros((alto, ancho), np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    if not np.any(mask):
        raise ValueError('Polígono sin superficie.')
    return mask


def guardar_calibracion(destino, fotograma, puntos_A, puntos_B):
    """Guarda fondo, homografía identidad y máscaras sin tocar datos históricos."""
    if fotograma is None or fotograma.ndim not in (2, 3):
        raise ValueError('Se necesita un fotograma real.')
    gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY) if fotograma.ndim == 3 else fotograma.copy()
    if gris.dtype != np.uint8:
        raise ValueError('El fotograma debe ser uint8.')
    ma = mascara_poligono(gris.shape, puntos_A)
    mb = mascara_poligono(gris.shape, puntos_B)
    if np.any(ma & mb):
        raise ValueError('Las ROI A y B se superponen. Dibuja calles sin superposición.')
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    np.save(destino / 'B.npy', gris)
    np.save(destino / 'H.npy', np.eye(3, dtype=np.float64))
    np.save(destino / 'mascara_A.npy', ma)
    np.save(destino / 'mascara_B.npy', mb)
    np.save(destino / 'puntos_A.npy', np.asarray(puntos_A, dtype=np.int32))
    np.save(destino / 'puntos_B.npy', np.asarray(puntos_B, dtype=np.int32))


class EditorCalibracion:
    def __init__(self, frame):
        self.frame = frame.copy()
        self.puntos = {'A': [], 'B': []}
        self.calle = 'A'
        self.fondo = None

    def clic(self, evento, x, y, flags, userdata):
        if evento == cv2.EVENT_LBUTTONDOWN:
            self.puntos[self.calle].append((x, y))

    def visualizar(self):
        vista = self.frame.copy()
        colores = {'A': (0, 220, 0), 'B': (255, 180, 0)}
        for calle in ('A', 'B'):
            pts = np.asarray(self.puntos[calle], np.int32)
            if len(pts):
                for p in pts:
                    cv2.circle(vista, tuple(p), 4, colores[calle], -1)
                if len(pts) >= 2:
                    cv2.polylines(vista, [pts], False, colores[calle], 2)
                if len(pts) >= 3:
                    overlay = vista.copy()
                    cv2.fillPoly(overlay, [pts], colores[calle])
                    vista = cv2.addWeighted(overlay, .18, vista, .82, 0)
        estado = 'LISTO' if self.fondo is not None else 'PENDIENTE'
        textos = [f'Editando calle {self.calle} | Fondo: {estado}',
                  'Mouse: anadir vertices | A/B: elegir calle | U: deshacer',
                  'F: guardar fotograma de fondo VACIO | G: guardar calibracion | Q: salir']
        h, w = vista.shape[:2]
        for i, t in enumerate(textos):
            y = 25 + 26 * i
            cv2.rectangle(vista, (0, y - 21), (min(w-1, 870), y+5), (18, 18, 18), -1)
            cv2.putText(vista, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1)
        return vista


def calibrar(cap, salida):
    """Editor gráfico. Presionar F con la escena vacía y G con 3+ puntos por ROI."""
    ok, frame = cap.read()
    if not ok:
        raise ValueError('No se pudo obtener el primer fotograma para calibrar.')
    editor = EditorCalibracion(frame)
    titulo = 'Calibracion A/B - pulsa F para guardar fondo vacio'
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(titulo, editor.clic)
    pausado = False
    try:
        while True:
            if not pausado:
                ok, nuevo = cap.read()
                if ok:
                    editor.frame = nuevo.copy()
                else:
                    pausado = True  # vídeos terminados: conserva el último fotograma
            cv2.imshow(titulo, editor.visualizar())
            tecla = cv2.waitKey(30) & 255
            if tecla in (ord('q'), 27):
                return False
            if tecla == ord(' '):
                pausado = not pausado
            elif tecla == ord('a'):
                editor.calle = 'A'
            elif tecla == ord('b'):
                editor.calle = 'B'
            elif tecla == ord('u'):
                if editor.puntos[editor.calle]:
                    editor.puntos[editor.calle].pop()
            elif tecla == ord('r'):
                editor.puntos[editor.calle] = []
            elif tecla == ord('f'):
                editor.fondo = editor.frame.copy()
                print('Fondo capturado. Debe ser una imagen SIN vehículos.')
            elif tecla == ord('g'):
                if editor.fondo is None:
                    print('Falta capturar un fondo sin vehículos: pulsa F.')
                    continue
                try:
                    guardar_calibracion(salida, editor.fondo, editor.puntos['A'], editor.puntos['B'])
                except ValueError as err:
                    print(f'Calibración no guardada: {err}')
                    continue
                print(f'Calibracion guardada en {Path(salida).resolve()}')
                return True
    finally:
        cv2.destroyWindow(titulo)
