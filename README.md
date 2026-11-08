# Detección de vehículos en calles A y B mediante Álgebra Lineal 🚗

**Proyecto académico de Álgebra Lineal — Ingeniería en Sistemas, Universidad Privada Domingo Savio (UPDS)**  
**Desarrollador de la implementación en Python:** Armando H. Mendoza  
**Tecnologías:** Python · NumPy · OpenCV · Visión artificial clásica

Este proyecto explora cómo las **matrices y las operaciones matemáticas sobre imágenes** pueden utilizarse para detectar *regiones candidatas a vehículos* en dos calles, denominadas **A** y **B**. Está orientado a demostrar aplicaciones de Álgebra Lineal en tecnología, no al control físico del tráfico.

> **Alcance:** no utiliza Arduino, servomotores, semáforos físicos ni modelos de inteligencia artificial entrenados. El contador muestra **candidatos visibles en cada fotograma**, no una identificación infalible de automóviles ni un conteo acumulado de vehículos que pasan.

## Características

- **Calibración desde cero:** dibuja por clic las regiones de interés de calle A y B para una nueva cámara o video.
- **Fondo de referencia:** captura un fotograma sin vehículos para compararlo con imágenes posteriores.
- **Procesamiento matricial:** grises, transformación de perspectiva y resta entre fotogramas y fondo.
- **Segmentación compartida:** un único procesamiento inicial y el mismo detector parametrizable aplicado a ambas calles.
- **Filtrado geométrico:** componentes conectados, área en píxeles, dimensiones, proporción, solidez y ocupación de la caja.
- **Modo diagnóstico:** inspección de píxeles activos y recuadros aceptados/rechazados.
- **Material académico original:** los scripts experimentales históricos están en `legacy/`.

## Estructura

```text
linear-algebra-traffic-vision/
├── main.py                    # Entrada: calibrar o detectar
├── src/
│   ├── calibracion.py         # Calibración interactiva A/B
│   ├── detector.py            # Segmentación y conteo compartido
│   └── vision_matrices.py     # Operaciones matemáticas originales
├── data/                      # Datos de calibración de la escena universitaria
├── legacy/                    # Scripts históricos conservados
├── tests/                     # Pruebas automáticas
├── docs/
│   ├── ORIGEN.md
│   └── MATEMATICAS.md
├── requirements.txt
└── README.md
```

## 1. Instalación en Ubuntu con VS Code

Abre **esta carpeta raíz** (donde está `main.py`) en VS Code. En la terminal:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si falta el módulo para crear entornos virtuales en Ubuntu, instala `python3-venv` mediante el gestor de paquetes de tu sistema.

Para verificar las dependencias:

```bash
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__, '| NumPy:', numpy.__version__)"
```

## 2. Ejecutar las pruebas

```bash
python -m unittest discover -s tests -v
```

Las pruebas evalúan casos sintéticos y aspectos de la calibración; **no validan la precisión con vehículos de un video real**.

## 3. Calibrar las calles A y B desde cero

Necesitas un video o una cámara **fija** que muestre ambas calles y, preferiblemente, un momento con las zonas vacías.

Con cámara:

```bash
python main.py --camera 0 --calibrar
```

Con video (sustituye la ruta por la de tu archivo):

```bash
python main.py --video /ruta/a/tu/video.mp4 --calibrar
```

**Controles de calibración:**

| Tecla / acción | Función |
| --- | --- |
| `Espacio` | Pausar o reanudar la imagen |
| `F` | Guardar el fondo de referencia **sin vehículos** |
| `A` / `B` | Elegir qué calle estás delimitando |
| Clic izquierdo | Agregar puntos al polígono (mínimo 3 por calle) |
| `U` | Deshacer el último punto de la calle seleccionada |
| `R` | Reiniciar el polígono de la calle seleccionada |
| `G` | Guardar la calibración |
| `Q` / `Esc` | Salir |

La calibración nueva se guarda en `calibracion_usuario/` y **no sobrescribe** los archivos históricos de `data/`. Los polígonos se dibujan directamente sobre la imagen; esta calibración usa una **matriz identidad**, sin rectificación de perspectiva.

## 4. Detectar en las calles A y B

Usa **la misma cámara, posición y resolución** que durante la calibración:

```bash
python main.py --camera 0 --data calibracion_usuario --diagnostico
```

O procesa el video correspondiente:

```bash
python main.py --video /ruta/a/tu/video.mp4 --data calibracion_usuario --diagnostico
```

En el modo diagnóstico, la máscara blanco/negro señala **píxeles diferentes del fondo**; los recuadros verdes/celestes representan candidatos aceptados en A/B y los rojos, regiones rechazadas.

Para ocultar la ventana adicional, omite `--diagnostico`. Para cerrar la ejecución, pulsa `Q` o `Esc`.

## 5. Ajustar filtros para reducir falsos positivos

```bash
python main.py --camera 0 --data calibracion_usuario --diagnostico \
  --umbral 45 --area-minima 500 --ancho-minimo 18 --alto-minimo 14
```

| Opción | Explicación |
| --- | --- |
| `--umbral` | Diferencia mínima de gris para activar un píxel |
| `--area-minima`, `--area-maxima` | Intervalo de área de las regiones detectadas |
| `--ancho-minimo`, `--alto-minimo` | Dimensiones mínimas del candidato |
| `--proporcion-min`, `--proporcion-max` | Intervalo de relación ancho/alto |
| `--solidez-min` | Compacidad según la envolvente convexa |
| `--ocupacion-min` | Fracción ocupada del rectángulo contenedor |
| `--kernel-cierre` | Une partes cercanas; un valor excesivo puede fusionar autos distintos |

Estos filtros **no son un clasificador semántico**. Una sombra, peatón u objeto semejante puede superar las reglas; varios vehículos superpuestos pueden aparecer como una sola región. Se requiere ajustar y verificar los parámetros sobre cada escena.

## 6. Fundamentos de Álgebra Lineal

Una imagen de grises puede entenderse como una matriz de intensidades, sobre la que se aplican operaciones como resta elemento a elemento, transformación proyectiva por una matriz 3×3, máscaras binarias y convoluciones. Los detalles y las ecuaciones están en **[Fundamentos matemáticos](docs/MATEMATICAS.md)**.

## Limitaciones y estado

- Demostración académica en evolución; no es una solución de gestión de tránsito para uso real.
- No utiliza modelos entrenados de IA ni identifica con certeza la clase automóvil.
- Cuenta candidatos **por fotograma**, sin seguimiento de IDs ni conteo acumulado de cruces.
- Requiere un fondo sin autos y **una cámara fija**; sombras y cambios de iluminación afectan la segmentación.
- La calibración original en `data/` corresponde a la escena universitaria; para otra escena, vuelve a calibrar.
- La exactitud del conteo **no está validada** con un conjunto representativo de videos reales.

## Procedencia

Esta implementación en Python es parte del trabajo académico realizado para Álgebra Lineal en la UPDS y posteriormente reorganizado para documentar su aplicación matemática. Los experimentos originales se conservan en `legacy/` para mostrar el proceso de desarrollo. No se incluyen funcionalidades de hardware.

## Licencia

No se incluye una licencia de reutilización por el momento. Si se decide publicar con una licencia abierta, deberá elegirse explícitamente.
