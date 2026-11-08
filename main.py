"""Calibración independiente de cámara/escena y detector de dos calles.
Sin Arduino ni dispositivos de control de tráfico.
"""
import argparse
from pathlib import Path
import cv2
from src.calibracion import calibrar
from src.detector import Parametros, cargar_calibracion, procesar_fotograma, dibujar_resultados

BASE = Path(__file__).resolve().parent


def ejecutar():
    p = argparse.ArgumentParser(description='Analizador visual de candidatos en calles A y B')
    origen = p.add_mutually_exclusive_group()
    origen.add_argument('--video', type=Path, help='Video de una escena, preferentemente con parte sin autos')
    origen.add_argument('--camera', type=int, default=0, help='Indice de cámara, por defecto 0')
    p.add_argument('--calibrar', action='store_true', help='Crear máscaras y fondo de escena nuevos')
    p.add_argument('--data', type=Path, default=BASE / 'data', help='Directorio de calibración para detección')
    p.add_argument('--salida-calibracion', type=Path, default=BASE / 'calibracion_usuario',
                   help='Directorio de salida para calibración nueva (por defecto no sobrescribe data)')
    p.add_argument('--umbral', type=int, default=40, help='Diferencia mínima de gris contra fondo')
    p.add_argument('--area-minima', type=int, default=300, help='Píxeles de objeto requeridos')
    p.add_argument('--area-maxima', type=int, default=0, help='0 significa sin máximo')
    p.add_argument('--kernel-cierre', type=int, default=9, help='Impar; para unir píxeles cercanos')
    p.add_argument('--ancho-minimo', type=int, default=12)
    p.add_argument('--alto-minimo', type=int, default=12)
    p.add_argument('--proporcion-min', type=float, default=.25)
    p.add_argument('--proporcion-max', type=float, default=5.0)
    p.add_argument('--solidez-min', type=float, default=.35)
    p.add_argument('--ocupacion-min', type=float, default=.32)
    p.add_argument('--diagnostico', action='store_true', help='Mostrar objetos rechazados y máscara')
    args = p.parse_args()
    fuente = str(args.video) if args.video else args.camera
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise SystemExit(f'No se puede abrir la fuente {fuente}. Prueba otro indice o archivo.')
    try:
        if args.calibrar:
            print('CALIBRACION: ESPACIO pausa, A/B cambia calle, clic puntos ROI, F fondo vacío, G guarda, Q sale')
            calibrar(cap, args.salida_calibracion)
            return
        fondo, H, mascaras = cargar_calibracion(args.data)
        parametros = Parametros(args.umbral, args.area_minima, args.area_maxima, args.kernel_cierre,
                               args.ancho_minimo, args.alto_minimo, args.proporcion_min,
                               args.proporcion_max, args.solidez_min, args.ocupacion_min)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            resultados, perspectiva, mascara = procesar_fotograma(
                frame, fondo, H, mascaras, parametros, diagnostico=args.diagnostico)
            vista = dibujar_resultados(perspectiva, mascara, mascaras, resultados, args.diagnostico)
            cv2.imshow('Detector de candidatos A/B - Q para salir', vista)
            if args.diagnostico:
                cv2.imshow('Pixeles activos - blanco', mascara * 255)
            if cv2.waitKey(25) & 255 in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ejecutar()
