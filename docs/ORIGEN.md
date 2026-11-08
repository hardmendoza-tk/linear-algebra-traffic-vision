
# Origen y evolución del proyecto

## Contexto académico

Este proyecto fue desarrollado en el contexto de la materia de Álgebra Lineal de la carrera de Ingeniería en Sistemas de la Universidad Privada Domingo Savio (UPDS).

**Desarrollador de la implementación en Python:** Armando H. Mendoza.

El objetivo es demostrar cómo los conocimientos de matrices y operaciones matemáticas pueden aplicarse al procesamiento digital de imágenes para detectar posibles vehículos en dos calles.

## Evolución del proyecto

### 1. Implementación académica original

Se desarrollaron experimentos relacionados con:

- Representación de imágenes mediante matrices.
- Conversión a escala de grises.
- Transformaciones de perspectiva.
- Operaciones de convolución.
- Resta de imágenes.
- Umbralización y máscaras binarias.
- Análisis de regiones de interés (ROI).

Los scripts experimentales históricos se conservan fuera de este repositorio.

### 2. Refactorización posterior

La implementación pública fue reorganizada para concentrarse en la detección de vehículos en las calles A y B.

Los módulos principales son:

- `src/vision_matrices.py`: operaciones matemáticas originales.
- `src/detector.py`: procesamiento compartido y detección de regiones candidatas.
- `src/calibracion.py`: calibración interactiva de las calles A y B.
- `main.py`: punto de entrada del programa.

## Calibración y datos

La carpeta `data/` contiene parámetros históricos de la escena universitaria.

Las nuevas calibraciones pueden guardarse en `calibracion_usuario/`, una carpeta local excluida de Git.

## Alcance actual

El proyecto utiliza visión artificial clásica basada en operaciones matemáticas.

No utiliza modelos de inteligencia artificial entrenados.

La detección se basa en diferencias de píxeles y características geométricas. Sus resultados deben interpretarse como estimaciones experimentales.
