"""
vision_matrices.py

Núcleo de procesamiento de imágenes y Álgebra Lineal
para el proyecto:

    Control Adaptativo de Tráfico mediante Visión por Computadora

Pilares matemáticos implementados:

1. Espacios/subespacios vectoriales:
       I ∈ R^(m×n)
       S_A, S_B mediante máscaras de ROI.

2. Sistemas de ecuaciones lineales:
       A · x = b
   para obtener la homografía mediante un sistema 8x8.

3. Determinante e inversa:
       det(H) != 0
       H^(-1)

4. Resta de matrices:
       D = |I - B|

5. Convolución:
       G * D
       Sobel X
       Sobel Y

6. Matriz binaria y norma/sumatoria L1:
       U ∈ {0,1}^(m×n)
       ||U||_1 = sum(U)
       P = sum(U) / número de píxeles ROI
"""

import cv2
import numpy as np

def obtener_camara(indice_pref=2, ancho=640, alto=480):
    """
    Abre el dispositivo de cámara. Intenta primero con el índice indicado 
    (/dev/video2 para DroidCam/S23) y si falla, recurre al índice 0.
    """
    cap = cv2.VideoCapture(indice_pref, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"Advertencia: No se pudo abrir /dev/video{indice_pref}. Intentando /dev/video0...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
    if not cap.isOpened():
        raise RuntimeError("No se pudo acceder a ninguna cámara disponible.")

    # Configurar la resolución recomendada
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    
    return cap

# ============================================================
# CAPTURA DE FOTOGRAMA
# ============================================================

def capturar_fotograma(cap):
    """
    Obtiene un fotograma de la cámara.

    Parámetros
    ----------
    cap : cv2.VideoCapture
        Objeto de captura de OpenCV.

    Retorna
    -------
    np.ndarray
        Fotograma BGR.

    None
        Si no fue posible capturar el fotograma.
    """

    if cap is None:
        raise ValueError("El objeto de cámara es None.")

    bLeido, aFrame = cap.read()

    if not bLeido:
        return None

    return aFrame


# ============================================================
# EXCEPCIONES PROPIAS
# ============================================================

class HomografiaSingularError(Exception):
    """Se produce cuando la homografía no puede invertirse."""
    pass


# ============================================================
# 1. ESPACIO VECTORIAL Y CONVERSIÓN A ESCALA DE GRISES
# ============================================================

def convertir_a_grises(frame):
    """
    Convierte una imagen BGR de OpenCV a escala de grises.

    Matemáticamente:

        I ∈ R^(m×n)

    En la implementación OpenCV utiliza normalmente uint8
    con valores entre 0 y 255.

    Parámetros
    ----------
    frame : np.ndarray
        Imagen BGR capturada por la cámara.

    Retorna
    -------
    np.ndarray
        Matriz I de escala de grises.
    """

    if frame is None:
        raise ValueError("El fotograma recibido es None.")

    if len(frame.shape) != 3:
        raise ValueError("El fotograma debe tener 3 canales BGR.")

    matriz_I = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return matriz_I


# ============================================================
# 1.2 MÁSCARAS DE ROI
# ============================================================

def crear_mascara_roi(shape, puntos):
    """
    Crea una máscara binaria para una Región de Interés.

    La máscara M tiene:

        M(i,j) = 1    si pertenece a la ROI
        M(i,j) = 0    si está fuera.

    Una ROI puede interpretarse como el resultado de aplicar
    una proyección/máscara:

        P_M(I) = I ⊙ M

    donde ⊙ representa multiplicación término a término.

    Parámetros
    ----------
    shape : tuple
        (alto, ancho) de la imagen.

    puntos : list / np.ndarray
        Coordenadas de los vértices del polígono.

    Retorna
    -------
    np.ndarray
        Máscara uint8 con valores 0 y 1.
    """

    alto, ancho = shape[:2]

    mascara = np.zeros((alto, ancho), dtype=np.uint8)

    puntos = np.array(puntos, dtype=np.int32)

    if len(puntos) < 3:
        raise ValueError("Una ROI necesita al menos 3 puntos.")

    cv2.fillPoly(mascara, [puntos], 1)

    return mascara


def proyectar_roi(matriz_I, mascara):
    """
    Proyecta una matriz I sobre la ROI.

        S_A = {I ⊙ M_A}

    o equivalentemente:

        P_A(I) = I ⊙ M_A

    Esta operación es completamente vectorizada con NumPy.
    """

    if matriz_I.shape != mascara.shape:
        raise ValueError(
            "La imagen y la máscara deben tener la misma dimensión."
        )

    return matriz_I * mascara


# ============================================================
# 2. SISTEMA A · x = b PARA HOMOGRAFÍA
# ============================================================

def construir_sistema_homografia(puntos_origen, puntos_destino):
    """
    Construye el sistema lineal:

        A · x = b

    con 8 ecuaciones y 8 incógnitas.

    La homografía se representa inicialmente como:

        H =
        [ h11 h12 h13 ]
        [ h21 h22 h23 ]
        [ h31 h32  1  ]

    El vector desconocido es:

        x =
        [h11, h12, h13,
         h21, h22, h23,
         h31, h32]^T

    Para cada correspondencia:

        (u,v) -> (x,y)

    se generan dos ecuaciones.

    Retorna
    -------
    A : np.ndarray
        Matriz 8x8.

    b : np.ndarray
        Vector 8x1.
    """

    puntos_origen = np.asarray(
        puntos_origen,
        dtype=np.float64
    )

    puntos_destino = np.asarray(
        puntos_destino,
        dtype=np.float64
    )

    if puntos_origen.shape != (4, 2):
        raise ValueError(
            "puntos_origen debe tener exactamente 4 puntos (4,2)."
        )

    if puntos_destino.shape != (4, 2):
        raise ValueError(
            "puntos_destino debe tener exactamente 4 puntos (4,2)."
        )

    A = []
    b = []

    for (u, v), (x, y) in zip(
        puntos_origen,
        puntos_destino
    ):

        # Ecuación asociada a X
        A.append([
            u, v, 1,
            0, 0, 0,
            -x * u,
            -x * v
        ])

        b.append(x)

        # Ecuación asociada a Y
        A.append([
            0, 0, 0,
            u, v, 1,
            -y * u,
            -y * v
        ])

        b.append(y)

    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    return A, b


def calcular_homografia(puntos_origen, puntos_destino):
    """
    Resuelve el sistema:

        A · x = b

    y construye:

        H =
        [h11 h12 h13]
        [h21 h22 h23]
        [h31 h32  1 ]

    Posteriormente:

        det(H) != 0

    permite obtener:

        H^(-1)

    Retorna
    -------
    H : np.ndarray
        Matriz de homografía 3x3.

    H_inversa : np.ndarray
        Matriz inversa 3x3.

    A : np.ndarray
        Sistema lineal 8x8.

    b : np.ndarray
        Vector independiente.
    """

    A, b = construir_sistema_homografia(
        puntos_origen,
        puntos_destino
    )

    # --------------------------------------------------------
    # Resolver A · x = b
    # --------------------------------------------------------

    try:

        # Verificación adicional del sistema.
        rango_A = np.linalg.matrix_rank(A)

        if rango_A < 8:
            raise HomografiaSingularError(
                "El sistema 8x8 no tiene rango completo."
            )

        vector_x = np.linalg.solve(A, b)

    except np.linalg.LinAlgError as error:

        raise HomografiaSingularError(
            "No fue posible resolver A · x = b. "
            "La matriz A puede ser singular."
        ) from error

    # --------------------------------------------------------
    # Construcción de H
    # --------------------------------------------------------

    H = np.array([
        [vector_x[0], vector_x[1], vector_x[2]],
        [vector_x[3], vector_x[4], vector_x[5]],
        [vector_x[6], vector_x[7], 1.0]
    ])

    # --------------------------------------------------------
    # 3. DETERMINANTE
    # --------------------------------------------------------

    determinante = np.linalg.det(H)

    # Umbral numérico dependiente de la escala de H.
    escala = max(1.0, np.linalg.norm(H, ord=2))

    tolerancia = (
        np.finfo(np.float64).eps
        * (escala ** 3)
        * 100
    )

    if not np.isfinite(determinante):
        raise HomografiaSingularError(
            "El determinante de H no es finito."
        )

    if abs(determinante) <= tolerancia:
        raise HomografiaSingularError(
            f"Homografía singular: det(H) = {determinante}"
        )

    # --------------------------------------------------------
    # 3. MATRIZ INVERSA
    # --------------------------------------------------------

    try:

        H_inversa = np.linalg.inv(H)

    except np.linalg.LinAlgError as error:

        raise HomografiaSingularError(
            "H no puede invertirse."
        ) from error

    return H, H_inversa, A, b


# ============================================================
# APLICAR HOMOGRAFÍA
# ============================================================

def transformar_perspectiva(frame, H, ancho_salida, alto_salida):
    """
    Aplica:

        p' = H · p

    mediante cv2.warpPerspective.

    La homografía ya debe haber sido calculada durante
    la calibración.

    IMPORTANTE:
    No se calcula H en cada frame porque sería innecesariamente
    costoso.
    """

    if H.shape != (3, 3):
        raise ValueError("H debe ser una matriz 3x3.")

    resultado = cv2.warpPerspective(
        frame,
        H,
        (ancho_salida, alto_salida)
    )

    return resultado


# ============================================================
# UTILIZAR H^(-1)
# ============================================================

def transformar_a_camara(punto, H_inversa):
    """
    Convierte un punto del espacio transformado de vuelta
    al sistema de coordenadas original de la cámara.

        p = H^(-1) · p'

    Parámetros
    ----------
    punto : tuple
        (x, y)

    H_inversa : np.ndarray
        Matriz H^(-1).

    Retorna
    -------
    tuple
        Coordenada transformada.
    """

    if H_inversa.shape != (3, 3):
        raise ValueError(
            "H_inversa debe ser una matriz 3x3."
        )

    x, y = punto

    punto_homogeneo = np.array(
        [x, y, 1.0],
        dtype=np.float64
    )

    resultado = H_inversa @ punto_homogeneo

    # Normalización de coordenadas homogéneas
    resultado = resultado / resultado[2]

    return (
        float(resultado[0]),
        float(resultado[1])
    )


# ============================================================
# 4. RESTA DE MATRICES
# ============================================================

def restar_matrices(matriz_I, matriz_B):
    """
    Calcula:

        D = |I - B|

    donde:

        I = imagen actual
        B = imagen de fondo

    La operación es término a término.

    OpenCV ejecuta esta operación de manera optimizada.
    """

    if matriz_I.shape != matriz_B.shape:
        raise ValueError(
            "I y B deben tener las mismas dimensiones."
        )

    D = cv2.absdiff(
        matriz_I,
        matriz_B
    )

    return D

# ============================================================
# MATRIZ DE FONDO
# ============================================================

def capturar_fondo(
    cap,
    H,
    ancho_salida,
    alto_salida,
    nMuestras=30
):
    """
    Captura la matriz de fondo B de la maqueta vacía.

    El fondo se calcula utilizando varias imágenes y la
    mediana de cada píxel:

        B(i,j) = mediana(I_1(i,j), I_2(i,j), ..., I_k(i,j))

    Utilizar varias muestras reduce pequeñas variaciones
    producidas por ruido de la cámara o iluminación.

    La imagen se transforma primero mediante H para que:

        B ∈ R^(m×n)

    pertenezca al mismo plano rectificado donde posteriormente
    analizaremos los vehículos.

    Parámetros
    ----------
    cap : cv2.VideoCapture
        Cámara abierta.

    H : np.ndarray
        Matriz de homografía 3x3.

    ancho_salida : int
        Ancho de la imagen rectificada.

    alto_salida : int
        Alto de la imagen rectificada.

    nMuestras : int
        Cantidad de fotogramas para construir B.

    Retorna
    -------
    np.ndarray
        Matriz de fondo B en escala de grises.
    """

    if nMuestras <= 0:
        raise ValueError(
            "nMuestras debe ser mayor que cero."
        )

    aMuestras = []

    for nIndice in range(nMuestras):

        bLeido, aFrame = cap.read()

        if not bLeido:
            raise RuntimeError(
                "No fue posible capturar un fotograma "
                "durante la calibración del fondo."
            )

        # ----------------------------------------------------
        # Corregir perspectiva
        # ----------------------------------------------------

        aVista_cenital = cv2.warpPerspective(
            aFrame,
            H,
            (ancho_salida, alto_salida)
        )

        # ----------------------------------------------------
        # Convertir a escala de grises
        # ----------------------------------------------------

        aGris = cv2.cvtColor(
            aVista_cenital,
            cv2.COLOR_BGR2GRAY
        )

        aMuestras.append(aGris)

        print(
            f"Capturando fondo: "
            f"{nIndice + 1}/{nMuestras}",
            end="\r"
        )

    print()

    # --------------------------------------------------------
    # Convertimos las muestras a una matriz tridimensional:
    #
    # (nMuestras, alto, ancho)
    # --------------------------------------------------------

    aMuestras = np.stack(
        aMuestras,
        axis=0
    )

    # --------------------------------------------------------
    # Mediana por cada posición (i,j)
    #
    # B(i,j) = mediana de las muestras
    # --------------------------------------------------------

    aFondo = np.median(
        aMuestras,
        axis=0
    )

    # Regresamos a uint8 para trabajar eficientemente
    # con OpenCV.
    aFondo = aFondo.astype(
        np.uint8
    )

    return aFondo

# ============================================================
# 5. CONVOLUCIÓN + GAUSSIANO + SOBEL
# ============================================================

def aplicar_convolucion(matriz_D):
    """
    Aplica:

    1. Filtro Gaussiano
    2. Sobel X
    3. Sobel Y
    4. Magnitud del gradiente

    Matemáticamente, la convolución local puede expresarse:

        (D * K)(i,j)

    donde K es un kernel.

    Sobel:

        Kx =
        [-1  0  1]
        [-2  0  2]
        [-1  0  1]

        Ky =
        [-1 -2 -1]
        [ 0  0  0]
        [ 1  2  1]

    La magnitud:

        G = sqrt(Gx² + Gy²)

    """

    if matriz_D is None:
        raise ValueError("La matriz D es None.")

    # --------------------------------------------------------
    # Conversión a float32 para cálculos numéricos
    # eficientes.
    # --------------------------------------------------------

    matriz_float = matriz_D.astype(
        np.float32,
        copy=False
    )

    # --------------------------------------------------------
    # Filtro Gaussiano
    #
    # Suaviza ruido antes del cálculo del gradiente.
    # --------------------------------------------------------

    matriz_suavizada = cv2.GaussianBlur(
        matriz_float,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # Kernel Sobel X
    # --------------------------------------------------------

    Kx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    # --------------------------------------------------------
    # Kernel Sobel Y
    # --------------------------------------------------------

    Ky = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    # --------------------------------------------------------
    # Convoluciones vectorizadas
    # --------------------------------------------------------

    Gx = cv2.filter2D(
        matriz_suavizada,
        cv2.CV_32F,
        Kx
    )

    Gy = cv2.filter2D(
        matriz_suavizada,
        cv2.CV_32F,
        Ky
    )

    # --------------------------------------------------------
    # Magnitud:
    #
    # G = sqrt(Gx² + Gy²)
    # --------------------------------------------------------

    magnitud = cv2.magnitude(
        Gx,
        Gy
    )

    return (
        matriz_suavizada,
        Gx,
        Gy,
        magnitud
    )


# ============================================================
# 6. UMBRALIZACIÓN Y FILTRADO MORFOLÓGICO
# ============================================================

def binarizar_matriz(matriz_gradiente, umbral):
    """
    Binariza y aplica operaciones morfológicas agresivas para unir 
    fragmentos de un mismo vehículo en un solo bloque sólido.
    """
    if matriz_gradiente is None:
        raise ValueError("La matriz del gradiente es None.")

    # Umbralización binaria
    _, U = cv2.threshold(
        matriz_gradiente,
        umbral,
        1,
        cv2.THRESH_BINARY
    )

    U = U.astype(np.uint8, copy=False)

    # Kernel de 7x7 para 'rellenar' los huecos entre techo, capó y sombras del auto
    kernel_cierre = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_apertura = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 1. Cierre morfológico: Fusiona las manchas cercanas de un mismo auto
    U_limpia = cv2.morphologyEx(U, cv2.MORPH_CLOSE, kernel_cierre)
    
    # 2. Apertura morfológica: Elimina píxeles aislados del pavimento
    U_limpia = cv2.morphologyEx(U_limpia, cv2.MORPH_OPEN, kernel_apertura)

    return U_limpia


# ============================================================
# CONTEO ROBUSTO POR COMPONENTES CONECTADAS (2D)
# ============================================================

def contar_vehiculos_cc(matriz_binaria, mascara_roi, area_minima=300, area_maxima=4000):
    """
    Cuenta vehículos ignorando manchitas pequeñas (ruido) y estimando
    solapamientos en autos grandes.
    """
    # Proyección Hadamard: U ⊙ M
    U_roi = cv2.bitwise_and(matriz_binaria, matriz_binaria, mask=mascara_roi)

    # Componentes conectadas (8-conectividad)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        U_roi, connectivity=8
    )

    n_autos = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Ignorar cualquier mancha menor a 300 píxeles
        if area >= area_minima:
            if area <= area_maxima:
                n_autos += 1
            else:
                # Si dos autos están pegados en un bloque gigante (ej. > 2000px)
                n_autos += int(round(area / 600.0))

    return n_autos
# ============================================================
# 6. NORMA / SUMATORIA L1 Y OCUPACIÓN
# ============================================================

def calcular_ocupacion(matriz_binaria, mascara_roi):
    """
    Calcula la densidad vehicular dentro de una ROI.

    Primero:

        U_A = U ⊙ M_A

    Luego:

        ||U_A||_1 = Σ U_A(i,j)

    Finalmente:

        P = ||U_A||_1 / N_A × 100

    donde N_A es la cantidad de píxeles pertenecientes
    a la ROI.

    Retorna
    -------
    porcentaje : float
        Porcentaje de ocupación.

    suma_l1 : int
        Sumatoria de los valores binarios.

    cantidad_pixeles : int
        Número de píxeles válidos de la ROI.
    """

    if matriz_binaria.shape != mascara_roi.shape:
        raise ValueError(
            "U y mascara_roi deben tener las mismas dimensiones."
        )

    # Proyección de U sobre la ROI.
    U_roi = matriz_binaria * mascara_roi

    # Norma L1 / sumatoria.
    suma_l1 = int(
        np.sum(U_roi, dtype=np.int64)
    )

    # Número de elementos pertenecientes a la ROI.
    cantidad_pixeles = int(
        np.count_nonzero(mascara_roi)
    )

    if cantidad_pixeles == 0:
        raise ValueError(
            "La ROI no contiene ningún píxel."
        )

    porcentaje = (
        suma_l1
        / cantidad_pixeles
        * 100.0
    )

    return (
        porcentaje,
        suma_l1,
        cantidad_pixeles
    )


# ============================================================
# CONTEO DE VEHÍCULOS
# ============================================================

def contar_vehiculos(matriz_binaria, mascara_roi, area_minima=80):
    """
    Cuenta componentes conectados aproximados mediante contornos.

    El conteo se realiza únicamente dentro de la ROI.

    IMPORTANTE:
    Este conteo es una primera versión. Si dos autos aparecen
    unidos en la matriz binaria, findContours puede interpretarlos
    como un solo objeto. Posteriormente podremos agregar:

        - apertura morfológica
        - cierre
        - separación de objetos
        - filtros por área

    sin modificar la arquitectura principal.
    """

    if matriz_binaria.shape != mascara_roi.shape:
        raise ValueError(
            "U y mascara_roi deben tener las mismas dimensiones."
        )

    # Aplicamos ROI.
    U_roi = (
        matriz_binaria
        * mascara_roi
    )

    # findContours trabaja normalmente con 0/255.
    imagen_contornos = (
        U_roi * 255
    ).astype(np.uint8)

    contornos, _ = cv2.findContours(
        imagen_contornos,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contador = 0

    for contorno in contornos:

        area = cv2.contourArea(contorno)

        if area >= area_minima:
            contador += 1

    return contador


# ============================================================
# MÉTRICAS COMPLETAS DE UNA CALLE
# ============================================================

def calcular_metricas_calle(
    matriz_binaria,
    mascara_roi,
    area_minima=80
):
    """
    Devuelve simultáneamente:

        - ocupación
        - suma L1
        - cantidad de píxeles ROI
        - cantidad aproximada de vehículos
    """

    (
        porcentaje,
        suma_l1,
        cantidad_pixeles
    ) = calcular_ocupacion(
        matriz_binaria,
        mascara_roi
    )

    vehiculos = contar_vehiculos(
        matriz_binaria,
        mascara_roi,
        area_minima
    )

    return {
        "ocupacion": porcentaje,
        "suma_L1": suma_l1,
        "pixeles_roi": cantidad_pixeles,
        "vehiculos": vehiculos
    }


# ============================================================
# SUBMATRIZ NUMÉRICA PARA LA INTERFAZ
# ============================================================

def obtener_submatriz(
    matriz,
    x,
    y,
    ancho=5,
    alto=5
):
    """
    Extrae una pequeña submatriz para mostrar en la UI.

    Esto nos permitirá enseñar, por ejemplo:

        [120 121 122 125 128]
        [119 120 121 126 130]
        [...]

    en tiempo real.

    La extracción es una operación pequeña y no afecta
    significativamente el rendimiento.
    """

    if matriz is None:
        raise ValueError(
            "La matriz es None."
        )

    alto_matriz, ancho_matriz = matriz.shape[:2]

    x_final = min(
        x + ancho,
        ancho_matriz
    )

    y_final = min(
        y + alto,
        alto_matriz
    )

    if x < 0 or y < 0:
        raise ValueError(
            "x e y no pueden ser negativos."
        )

    submatriz = matriz[
        y:y_final,
        x:x_final
    ]

    return submatriz.copy()


# ============================================================
# FUNCIÓN DE DIAGNÓSTICO
# ============================================================

def informacion_matriz(matriz, nombre="Matriz"):
    """
    Devuelve información útil para mostrar o depurar:

        dimensión
        tipo
        mínimo
        máximo
        suma
    """

    if matriz is None:
        raise ValueError(
            f"{nombre} es None."
        )

    return {
        "nombre": nombre,
        "dimension": matriz.shape,
        "dtype": str(matriz.dtype),
        "minimo": float(np.min(matriz)),
        "maximo": float(np.max(matriz)),
        "suma": float(np.sum(matriz))
    }
def contar_vehiculos_proyeccion(U_binaria, mascara, umbral_fila=10, longitud_minima_auto=12, eje_proyeccion=1):
    """
    Cuenta vehículos proyectando la matriz binaria sobre un vector.
    - eje_proyeccion = 1: Suma por filas (axis=1). Para calles verticales.
    - eje_proyeccion = 0: Suma por columnas (axis=0). Para calles horizontales.
    """
    # Aplicar la máscara ROI mediante multiplicación término a término (U ⊙ M)
    U_roi = cv2.bitwise_and(U_binaria, U_binaria, mask=mascara)
    
    # Proyección matricial (eje 1 = filas, eje 0 = columnas)
    vector_perfil = np.sum(U_roi, axis=eje_proyeccion)
    
    # Binarizar el vector de presencia
    presencia = (vector_perfil > umbral_fila).astype(np.uint8)
    
    # Detectar bloques de presencia en el vector
    diferencias = np.diff(np.pad(presencia, (1, 1), 'constant'))
    inicios = np.where(diferencias == 1)[0]
    fines = np.where(diferencias == -1)[0]
    
    n_autos = 0
    for ini, fin in zip(inicios, fines):
        if (fin - ini) >= longitud_minima_auto:
            n_autos += 1
            
    return n_autos