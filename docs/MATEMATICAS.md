# Fundamentos matemáticos del proyecto

## 1. Imágenes como matrices

Una imagen en escala de grises puede representarse como una matriz $I \in \{0,\ldots,255\}^{H\times W}$. Cada entrada $I(y,x)$ corresponde a la intensidad de un píxel. En color, la imagen contiene canales adicionales.

## 2. Transformación de perspectiva

La homografía se representa por una matriz $H\in\mathbb{R}^{3\times3}$ y transforma coordenadas homogéneas:

$$
\begin{bmatrix}x'\\y'\\w'\end{bmatrix}
=H\begin{bmatrix}x\\y\\1\end{bmatrix},\qquad
(x_{salida},y_{salida})=\left(\frac{x'}{w'},\frac{y'}{w'}\right).
$$

Los datos históricos permiten trabajar con una homografía de escena. **La calibración interactiva nueva usa $H=I_3$** (matriz identidad), porque las regiones se dibujan directamente sobre el fotograma.

## 3. Resta de matrices

El sistema compara la imagen actual $I_t$ con una referencia del fondo $B$:

$$D_t(x,y)=\left|I_t(x,y)-B(x,y)\right|.$$

La resta mide cambios de intensidad. La referencia debe representar la misma escena y encuadre.

## 4. Umbralización y máscaras

Con un umbral $T$, la máscara binaria es:

$$M_t(x,y)=\begin{cases}1 & D_t(x,y)>T\\0 & \text{en otro caso.}\end{cases}$$

Cada calle dispone de una máscara ROI $R_A$ o $R_B$. La imagen de cambios se restringe mediante la intersección lógica $M_t\land R_k$ para $k\in\{A,B\}$.

## 5. Filtros espaciales y operaciones morfológicas

Un filtro Gaussiano reduce variaciones de alta frecuencia. Las operaciones morfológicas de cierre y apertura conectan fragmentos próximos o eliminan pequeñas regiones de ruido. La elección del tamaño del kernel afecta directamente al resultado.

## 6. Componentes conectados y geometría

Sobre la máscara binaria, OpenCV identifica componentes conectados. Se calculan área, ancho, alto, relación $w/h$, solidez del contorno y proporción del rectángulo ocupada. Estas **reglas geométricas** aceptan o rechazan candidatos, sin aprendizaje automático.

La ocupación de una calle puede expresarse como:

$$O_k=100\frac{\#\{(x,y): M_t(x,y)=1 \land R_k(x,y)=1\}}{\#\{(x,y): R_k(x,y)=1\}}.$$

El número de componentes aceptados es un conteo **instantáneo de candidatos**, no una medición validada de vehículos reales ni de tráfico acumulado.

## Relación con la materia

Este prototipo ilustra cómo matrices, transformaciones lineales/proyectivas, operaciones sobre arreglos y análisis geométrico se utilizan en aplicaciones tecnológicas de visión artificial. La detección utiliza operaciones matemáticas clásicas y no incluye una red neuronal ni un modelo preentrenado.
