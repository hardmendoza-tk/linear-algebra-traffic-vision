"""Procesamiento compartido de las calles A/B y filtros geométricos auditables.

Detecta regiones distintas del fondo que parecen objetos. No clasifica vehículos
semánticamente: una sombra u otro objeto con silueta similar puede pasar el filtro.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .vision_matrices import convertir_a_grises, transformar_perspectiva, restar_matrices


@dataclass(frozen=True)
class Parametros:
    umbral: int = 40
    area_minima: int = 300
    area_maxima: int = 0
    kernel_cierre: int = 9
    ancho_minimo: int = 12
    alto_minimo: int = 12
    proporcion_min: float = 0.25
    proporcion_max: float = 5.0
    solidez_min: float = 0.35
    ocupacion_min: float = 0.32


def cargar_calibracion(carpeta: Path):
    carpeta = Path(carpeta)
    fondo = np.load(carpeta / 'B.npy', allow_pickle=False)
    homografia = np.load(carpeta / 'H.npy', allow_pickle=False)
    mascaras = {calle: np.load(carpeta / f'mascara_{calle}.npy', allow_pickle=False)
                for calle in ('A', 'B')}
    if fondo.ndim != 2 or fondo.dtype != np.uint8:
        raise ValueError('B.npy debe ser una imagen gris uint8 2D.')
    if homografia.shape != (3, 3) or not np.isfinite(homografia).all():
        raise ValueError('H.npy debe ser una homografía finita 3x3.')
    if abs(np.linalg.det(homografia)) < 1e-12:
        raise ValueError('H.npy es singular.')
    for calle, mascara in mascaras.items():
        if mascara.shape != fondo.shape or not np.any(mascara):
            raise ValueError(f'Mascara calle {calle} incompatible o vacía.')
        mascaras[calle] = (mascara > 0).astype(np.uint8)
    return fondo, homografia, mascaras


def detectar_componentes(binaria, mascara_roi, params=Parametros(), diagnostico=False):
    """Encuentra un candidato por componente conectado que supera filtros.

    Filtros: píxeles activos mínimos/máximos, dimensiones, relación ancho/alto,
    solidez del contorno y ocupación de la caja. Aceptar una región NO garantiza
    que sea un automóvil, ni equivale a contar autos que cruzaron la carretera.
    """
    if binaria.shape != mascara_roi.shape:
        raise ValueError('La imagen y ROI deben tener el mismo tamaño.')
    if not 0 <= params.umbral <= 255 or params.area_minima < 1:
        raise ValueError('Umbral o area minima invalidos.')
    if params.area_maxima and params.area_maxima < params.area_minima:
        raise ValueError('Area maxima menor que minima.')
    if params.kernel_cierre < 1 or params.kernel_cierre % 2 == 0:
        raise ValueError('Kernel de cierre debe ser positivo e impar.')
    if not (0 < params.proporcion_min <= params.proporcion_max and
            0 <= params.solidez_min <= 1 and 0 <= params.ocupacion_min <= 1):
        raise ValueError('Parametros geometricos invalidos.')
    roi = (mascara_roi > 0).astype(np.uint8)
    fg = ((binaria > 0) & (roi > 0)).astype(np.uint8)
    if params.kernel_cierre > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.kernel_cierre, params.kernel_cierre))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg *= roi
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    aceptados, rechazados = [], []
    for i in range(1, n):
        x, y, w, h, area = map(int, stats[i])
        if w < params.ancho_minimo or h < params.alto_minimo:
            motivo = 'dimensiones'
        elif area < params.area_minima:
            motivo = 'area_min'
        elif params.area_maxima and area > params.area_maxima:
            motivo = 'area_max'
        elif not (params.proporcion_min <= w / h <= params.proporcion_max):
            motivo = 'proporcion'
        else:
            objeto = np.uint8(labels[y:y+h, x:x+w] == i) * 255
            contornos, _ = cv2.findContours(objeto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos:
                motivo = 'sin_contorno'
            else:
                contorno = max(contornos, key=cv2.contourArea)
                hull_area = cv2.contourArea(cv2.convexHull(contorno))
                solidez = cv2.contourArea(contorno) / max(hull_area, 1e-6)
                ocupacion_caja = area / (w * h)
                if solidez < params.solidez_min:
                    motivo = 'solidez'
                elif ocupacion_caja < params.ocupacion_min:
                    motivo = 'ocupacion_caja'
                else:
                    motivo = None
        candidato = (x, y, w, h, area)
        if motivo is None:
            aceptados.append(candidato)
        else:
            rechazados.append((candidato, motivo))
    aceptados.sort(key=lambda c: (c[1], c[0]))
    ocupacion_roi = 100.0 * np.count_nonzero(fg) / np.count_nonzero(roi)
    if diagnostico:
        return aceptados, float(ocupacion_roi), fg, rechazados
    return aceptados, float(ocupacion_roi), fg


def procesar_fotograma(frame, fondo, homografia, mascaras, params=Parametros(), diagnostico=False):
    if frame is None:
        raise ValueError('No se recibió un fotograma.')
    alto, ancho = fondo.shape
    gris = convertir_a_grises(frame) if frame.ndim == 3 else frame
    perspectiva = transformar_perspectiva(gris, homografia, ancho, alto)
    diferencia = restar_matrices(perspectiva, fondo)
    diferencia = cv2.GaussianBlur(diferencia, (5, 5), 0)
    _, binaria = cv2.threshold(diferencia, params.umbral, 1, cv2.THRESH_BINARY)
    resultados = {}
    mascara_limpia = np.zeros_like(binaria, dtype=np.uint8)
    for calle in ('A', 'B'):
        encontrados = detectar_componentes(binaria, mascaras[calle], params, diagnostico=diagnostico)
        if diagnostico:
            candidatos, porcentaje, mascara, rechazados = encontrados
        else:
            candidatos, porcentaje, mascara = encontrados
        mascara_limpia = np.maximum(mascara_limpia, mascara)
        resultados[calle] = {'vehiculos_aprox': len(candidatos), 'ocupacion_pct': porcentaje,
                             'candidatos': candidatos}
        if diagnostico:
            resultados[calle]['rechazados'] = rechazados
    return resultados, perspectiva, mascara_limpia


def dibujar_resultados(imagen_gris, binaria, mascaras, resultados, diagnostico=False):
    panel = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    colores = {'A': (20, 210, 30), 'B': (245, 170, 40)}
    for i, calle in enumerate(('A', 'B')):
        contornos, _ = cv2.findContours((mascaras[calle] * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contornos, -1, colores[calle], 2)
        d = resultados[calle]
        for x, y, w, h, area in d['candidatos']:
            cv2.rectangle(panel, (x, y), (x+w, y+h), colores[calle], 2)
            cv2.putText(panel, str(area), (x, max(y-3, 8)), cv2.FONT_HERSHEY_SIMPLEX, .38, colores[calle], 1)
        if diagnostico:
            for ((x, y, w, h, _), _) in d.get('rechazados', []):
                cv2.rectangle(panel, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.putText(panel, f'Calle {calle}: {len(d["candidatos"])} candidatos | {d["ocupacion_pct"]:.1f}% ROI',
                    (10, 25 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, .55, colores[calle], 2)
    return panel
