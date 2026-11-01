# Documentación técnica

Este documento explica **cómo funciona por dentro** cada uno de los dos programas del proyecto, qué técnicas de visión artificial se usan, **por qué** se eligieron y qué hace cada función. Para instalar y ejecutar el proyecto, consulta el [README](README.md).

## Índice

1. [Visión general](#1-visión-general)
2. [Técnicas de visión usadas (resumen)](#2-técnicas-de-visión-usadas-resumen)
3. [Parte A: detector de cartas](#3-parte-a-detector-de-cartas)
4. [Parte B: detector de señas](#4-parte-b-detector-de-señas)
5. [Cómo extender el proyecto](#5-cómo-extender-el-proyecto)
6. [Glosario](#6-glosario)

---

## 1. Visión general

El proyecto resuelve dos problemas distintos con las mismas herramientas (Python + OpenCV + NumPy) y la misma filosofía: **visión clásica, sin entrenar redes neuronales**. Cada paso del procesado se puede ver en una ventana de depuración y se puede explicar con una operación sobre píxeles.

| | Cartas | Señas |
|---|---|---|
| **Entrada** | Cámara del móvil mirando a la mesa | Webcam mirando a la cara |
| **Problema** | Reconocer números impresos | Reconocer gestos faciales |
| **Idea clave** | Aislar la carta, enderezarla y comparar su número con moldes | Encontrar la cara y analizar zonas fijas (ojos, cejas, boca) |
| **Salida** | Número de cada carta + jugadas del Mus | Texto de la seña detectada |

Ambos siguen la **misma arquitectura en capas**:

```
 config_*.py   → constantes y umbrales (lo que se ajusta)
 detector_*.py → motor de visión (procesa píxeles, no toca la cámara)
 main_*.py     → abre la cámara, une las piezas y dibuja el resultado
```

Separar así el código permite cambiar un umbral sin buscar en 300 líneas, probar el motor sin cámara y mantener las reglas del Mus (`logica_mus.py`) totalmente aisladas de la visión.

---

## 2. Técnicas de visión usadas (resumen)

| Técnica | Función de OpenCV | Para qué se usa aquí | Por qué esta técnica |
|---|---|---|---|
| Escala de grises | `cvtColor(..., COLOR_BGR2GRAY)` | Quitar el color antes de umbralizar | Menos datos y la forma importa más que el color |
| Desenfoque gaussiano | `GaussianBlur` | Eliminar ruido de cámara y texturas | Sin él, el ruido crea manchas falsas al umbralizar |
| Umbral de Otsu | `threshold(..., THRESH_OTSU)` | Separar tinta y papel en la esquina de la carta | Calcula solo el umbral óptimo según el histograma |
| Umbral adaptativo | `adaptiveThreshold` | Encontrar cartas en la mesa; segunda binarización del número | Calcula un umbral por zona, tolera luz desigual |
| Umbral fijo | `threshold(..., 100)` | Tercera binarización de respaldo | Funciona cuando Otsu falla por una mancha grande |
| Morfología (cierre) | `morphologyEx(MORPH_CLOSE)` | Rellenar huecos y que cada carta sea un bloque sólido | Evita que el contorno parpadee |
| Morfología (apertura con kernel largo) | `morphologyEx(MORPH_OPEN)` | Borrar líneas rectas (marco de la carta); en señas, borrar labios y pelos | Un kernel de 45 px solo conserva líneas de más de 45 px |
| Contornos | `findContours`, `contourArea` | Localizar cartas y manchas | Da forma, área y posición de cada objeto |
| Rectángulo mínimo rotado | `minAreaRect`, `boxPoints` | Obtener las 4 esquinas de una carta girada | Funciona para cualquier ángulo |
| Transformación de perspectiva | `getPerspectiveTransform`, `warpPerspective` | "Planchar" cada carta a 400x600 | Todas las cartas quedan con el mismo tamaño y orientación |
| Filtro de nitidez | `filter2D` | Compensar desenfoque | Resalta la tinta del número |
| Componentes conectadas | `connectedComponentsWithStats` | Separar cada mancha de tinta con su caja y área | Más directo que contornos cuando se necesitan las estadísticas |
| *Template matching* | `matchTemplate(TM_CCOEFF_NORMED)` | Comparar el número con los moldes | Rápido y fiable con solo 10 clases y una tipografía fija |
| Seguimiento por distancia + moda | `math.hypot`, `Counter` | Estabilizar el número de cada carta | Un fotograma malo no cambia la lectura |
| Clasificadores Haar (Viola-Jones) | `CascadeClassifier.detectMultiScale` | Detectar cara y ojos | Preentrenados, incluidos en OpenCV |
| CLAHE | `createCLAHE` | Igualar la luz local de la cara | Ayuda a Haar y a los umbrales con luz desigual |
| Espacio de color HSV | `cvtColor(BGR2HSV)`, `inRange` | Aislar el color de la lengua | El tono (H) separa el color del brillo, más robusto que RGB |
| Media móvil | listas con los últimos N valores | Suavizar la caja de la cara | Quita el temblor |
| Máquina de estados con contadores | variables `contador_*` | Confirmar señas | Evita falsos positivos por un solo fotograma |

---

## 3. Parte A: detector de cartas

### 3.1 Objetivo y reglas del juego

El objetivo es leer una mano de **4 cartas** de la baraja española y decir qué jugadas contiene. Solo se necesita el **número**, no el palo.

El programa asume las reglas de **8 reyes**:

- El **3** cuenta como **rey** (12) y el **2** como **as** (1) en todas las jugadas.
- Las figuras (10, 11, 12) valen **10 puntos** para el juego; las demás, su número.

Jugadas que evalúa:

| Jugada | Regla |
|---|---|
| **Pares** | *Duples*: cuatro iguales o dos parejas. *Medias*: tres iguales. *Par*: dos iguales. |
| **Juego** | Suma ≥ 31. El 31 es la mejor puntuación (`Max!`). Con menos de 31 se informa el **punto** (la suma). |
| **Grandes** | Las dos cartas más altas de la mano. |
| **Chicas** | Las dos cartas más bajas de la mano. |

### 3.2 Archivos de esta parte

| Archivo | Rol |
|---|---|
| `main_cartas.py` | Programa principal: cámara, bucle de vídeo, seguimiento y HUD |
| `config_cartas.py` | Constantes: rutas, cámara, tamaños y tracking |
| `detector_cartas.py` | Motor de visión: cargar moldes, extraer el número y reconocerlo |
| `logica_mus.py` | Reglas del Mus, independientes de la visión |
| `generador_moldes.py` | Herramienta offline para crear los moldes |
| `moldes/` | Moldes definitivos `molde_N.png` |
| `conexion_camara.py` | Prueba de cámara y FPS |
| `mus_directo.py` | Versión monolítica anterior (referencia histórica) |

### 3.3 Visión global: dos fases

El sistema trabaja en dos momentos diferentes:

```
 FASE 1 (una sola vez, offline)             FASE 2 (en directo, cada fotograma)
 ───────────────────────────────            ────────────────────────────────────
 baraja.jpg                                  Cámara
    │                                          │
 generador_moldes.py                          main_cartas.py
    │  detecta cartas, lee la esquina          │  detecta cartas, las endereza
    ▼                                          ▼
 moldes_generados/  ──(renombrar a mano)──▶ moldes/ ──▶ detector_cartas.reconocer_numero
                                                              │
                                                          logica_mus.evaluar_mus
                                                              │
                                                          Pantalla (HUD)
```

### 3.4 Fase 1: generar los moldes (`generador_moldes.py`)

Un **molde** es una imagen binaria de 60x80 píxeles con el número en blanco sobre fondo negro. Se generan una vez a partir de una foto de la baraja y después se comparan con las cartas en directo.

Pasos del script:

1. **Cargar y redimensionar** `baraja.jpg` a 1200 px de ancho (las fotos del móvil son enormes y no caben en pantalla).
2. **Encontrar las cartas**: grises → desenfoque gaussiano 5x5 → umbral adaptativo (bloque 21, constante 4) → `findContours` externos. Se aceptan los contornos con área entre 5000 y 300000.
3. **Planchar cada carta** a 400x600 con `getPerspectiveTransform` + `warpPerspective`, tras ordenar las 4 esquinas con `ordenar_puntos`.
4. **Extraer el número** con `extraer_numero` (ver [3.6](#36-motor-de-visión-detector_cartaspy)).
5. **Guardar** el molde en `moldes_generados/carta_generada_NN.png` y mostrar ventanas con cada paso (grises, suavizado, binarizada, contornos y las cartas planchadas, con el recuadro verde donde se leyó el número).
6. **Paso manual:** se comprueba cada imagen, se renombra con el valor real (`molde_5.png`…) y se copia a `moldes/`. Se hace a mano porque el orden en que salen las cartas depende del orden de los contornos, no del valor de la carta.

> Este script es **autónomo a propósito**: repite las funciones de `detector_cartas.py` para que se pueda leer de principio a fin como una demostración del método. Es la versión "didáctica" de la extracción del número.
>
> Pequeña diferencia: `buscar_digito` del generador tiene un filtro extra (la **prueba del anillo**: un dígito está rodeado de papel, mientras que los restos de mesa o marco están rodeados de tinta), que no se usa en el motor en directo.

### 3.5 Fase 2: el bucle en directo (`main_cartas.py`)

La función `iniciar_mus_directo()` ejecuta estos pasos en cada fotograma:

#### Paso 0: inicialización

- Se cargan los moldes en memoria con `cargar_diccionario_moldes()` (diccionario `{número: imagen}`).
- Se abre la cámara con la resolución de `config_cartas.py` (1920x1080) y `CAP_PROP_BUFFERSIZE = 1`. Este último parámetro es el truco **anti-lag**: obliga a OpenCV a guardar solo el último fotograma, así no se procesan imágenes viejas.

#### Paso A: encontrar las siluetas de las cartas

```
frame → gris → GaussianBlur(15x15) → adaptiveThreshold(51, 10) → CLOSE(9x9) → findContours
```

- **Desenfoque 15x15:** es muy agresivo a propósito. Mata el ruido y la textura de la mesa, y deja solo los cambios grandes de brillo (carta contra mesa).
- **Umbral adaptativo con bloque 51 y constante 10:** marca como blanco lo que es más oscuro que su entorno. Un bloque grande ignora las microsombras. Funciona con luz desigual porque el umbral se calcula por zonas.
- **Cierre morfológico 9x9:** rellena los huecos que el ruido deja en el contorno y convierte cada carta en un bloque sólido. Evita que el recuadro verde parpadee.

Este método funciona mejor con **mesa oscura**: la diferencia de brillo entre carta y mesa es grande y estable.

#### Paso B: filtrar candidatos

Para cada contorno se aplican dos filtros:

1. **Área** entre `AREA_MINIMA` y `AREA_MAXIMA` (descarta ruido, y objetos enormes como la propia mesa).
2. **Proporción:** se calcula `lado largo / lado corto` del `minAreaRect` y se rechaza si supera `ASPECT_RATIO_MAX` (2.5). Evita lápices, brazos o móviles. Una carta real tiene ≈ 1.5.

#### Paso C: enderezar ("planchar") la carta

1. `boxPoints` da las 4 esquinas del rectángulo mínimo rotado.
2. `ordenar_puntos` las coloca siempre en el orden *arriba-izquierda, arriba-derecha, abajo-derecha, abajo-izquierda* (sumando y restando coordenadas). Sin este orden, la carta saldría girada o espejada.
3. **Parche anti-deformación:** si el lado superior es más largo que el izquierdo, la carta está **tumbada**. Se rotan las esquinas con `np.roll` para que el lado corto quede arriba. Así el número no se aplasta ni se estira.
4. `getPerspectiveTransform` + `warpPerspective` proyectan la carta a un lienzo fijo de **400x600**. A partir de aquí, todas las cartas tienen el mismo tamaño y el número siempre cae en el mismo sitio, lo que permite usar filtros con valores fijos en píxeles.

#### Paso D: filtro de nitidez

Se aplica un kernel de *sharpening* `[[0,-1,0],[-1,5,-1],[0,-1,0]]` con `filter2D`. Compensa el desenfoque ligero de la cámara y realza la tinta del número.

#### Paso E: extraer el número y reconocerlo

- `extraer_numero(carta)` devuelve el número aislado como imagen binaria de 60x80.
- `reconocer_numero(molde_vivo, diccionario)` lo compara con los moldes y devuelve el valor (1…12) o `None`.

Detalles en [3.6](#36-motor-de-visión-detector_cartaspy).

#### Paso F: memoria espacial (anti-parpadeo)

Un solo fotograma puede leer mal un número por una sombra o un reflejo. Para evitarlo, el programa **recuerda cada carta** entre fotogramas.

`memoria_mesa` es una lista de diccionarios, uno por carta:

```python
{'cx': ..., 'cy': ...,   # centro de la carta en el último fotograma
 'historial': [...],     # últimas lecturas del número
 'vida': 10}             # fotogramas que quedan antes de olvidarla
```

Para cada carta detectada:

1. Se busca en la memoria una carta con el centro a menos de `TRACKING_DISTANCIA_MAX` (100 px). Si existe, es **la misma carta**: se actualiza su posición y se rellena su `vida`.
2. Si la lectura actual no es `None`, se añade al `historial` (máximo `TRACKING_HISTORIAL_MAX` = 15 valores).
3. El **número estable** es la **moda** del historial (`Counter(...).most_common(1)`): el valor que más veces se ha leído.
4. Si no hay ninguna carta cercana en memoria, es una carta nueva y se registra.

Al final de cada fotograma, la `vida` de todas las cartas baja en 1 y se borran las que llegan a 0. Así, si quitas una carta de la mesa, desaparece de la memoria en ~10 fotogramas, y si la cámara pierde una carta un instante, no se olvida de golpe.

#### Paso G: dibujar y evaluar

- Solo se dibujan (recuadro verde + número amarillo) las cartas con número estable.
- Cada texto se dibuja dos veces: primero en negro y más grueso, después en color encima. La "sombra" hace que se lea sobre cualquier fondo.
- Con **4 o más** cartas leídas se llama a `mus.evaluar_mus(...)` y se imprimen las líneas del resultado en cascada. Si faltan cartas, se muestra `Esperando 4 cartas... (Detectadas: N)` en rojo.
- Arriba a la izquierda se muestran los FPS y la lista `Mesa: [...]`.

### 3.6 Motor de visión (`detector_cartas.py`)

Este módulo solo procesa píxeles. No abre la cámara ni sabe nada del Mus.

#### Cómo se localiza el número: la idea

En cada carta española, el número está impreso **dos veces**: arriba a la izquierda y, boca abajo, abajo a la derecha. Como la carta ya está planchada a 400x600, el número cae siempre en la esquina superior izquierda, en una zona de unos 170x170 px. El algoritmo:

1. Recorta esa esquina.
2. Binariza (tinta = blanco).
3. Borra el marco de la carta.
4. Busca manchas de tinta y descarta las que no parecen un dígito.
5. Si no encuentra nada, **gira la carta 180°** y repite (así lee la esquina inferior derecha).

#### Referencia de funciones

| Función | Qué hace |
|---|---|
| `cargar_diccionario_moldes()` | Lee `moldes/molde_N.png`, extrae `N` del nombre, los binariza (umbral 127) y devuelve `{N: imagen}`. Termina el programa si no existe la carpeta. |
| `reconocer_numero(molde_extraido, diccionario)` | Compara el número leído con todos los moldes y devuelve el más parecido, o `None` si la similitud no supera 0.55. |
| `ordenar_puntos(puntos)` | Ordena 4 esquinas: arriba-izq (menor `x+y`), abajo-der (mayor `x+y`), arriba-der (menor `y-x`), abajo-izq (mayor `y-x`). |
| `quitar_lineas(binaria)` | Borra las líneas rectas largas (marco de la carta). |
| `buscar_digito(binaria)` | Encuentra y recorta el dígito en una esquina ya binarizada; devuelve `(imagen 60x80, caja)` o `(None, None)`. |
| `extraer_numero(carta)` | Función principal: prueba 2 esquinas × 3 binarizaciones y devuelve el primer resultado válido. |

#### `extraer_numero`: esquinas y binarizaciones

Para cada orientación (normal, y girada 180°) recorta `carta[0:170, 0:170]`, la pasa a gris y la suaviza (gaussiano 3x3). Luego prueba **tres binarizaciones** distintas y se queda con la primera que dé un dígito:

| # | Método | Cuándo es útil |
|---|---|---|
| 1 | Otsu invertido | Caso normal: separa solo tinta y papel |
| 2 | Adaptativo gaussiano (bloque 25, C=10) | Iluminación desigual o sombras |
| 3 | Umbral fijo 100 | Cuando una mancha grande engaña a Otsu |

Tener **respaldos** hace al sistema mucho más robusto: no depende de que un único umbral acierte siempre.

#### `quitar_lineas`: borrar el marco

Las cartas tienen un marco negro rectangular. En ciertas cartas el número queda pegado a él. Una **apertura morfológica** con un kernel vertical de 1x45 y otro horizontal de 45x1 solo conserva las líneas rectas de más de 45 px. Se juntan, se dilatan 3x3 y se restan de la imagen. El número (curvo y de ~25-30 px de alto) no sobrevive a esa apertura, así que no se borra.

#### `buscar_digito`: filtrar manchas

Tras limpiar las líneas, `connectedComponentsWithStats` devuelve cada mancha con su caja `(x, y, w, h)` y su área. Cada mancha debe superar **todos** estos filtros:

| Filtro | Condición | Qué elimina |
|---|---|---|
| Tamaño | `4 < w < 60`, `18 < h < 60`, `área > 60` | Polvo y dibujos grandes |
| Borde | No toca el borde del recorte | Manchas cortadas (trozos de mesa o marco) |
| Esquina | Centro con `x ≥ 35` e `y ≥ 35` | El arco redondeado de la esquina de la carta |

Con las manchas supervivientes:

1. **Ancla:** se ordenan por `x + y`; la más cercana a la esquina es el primer dígito.
2. **Agrupación:** se unen a ella las manchas a menos de 50 px en horizontal y con el centro a menos de 15 px de altura. Así los números de dos cifras (10, 11, 12) forman un solo grupo.
3. Se calcula la caja que envuelve al grupo y se añade un **margen de 4 px**.
4. Se recorta de la imagen sin líneas, y dentro del recorte se borran las manchas de menos de **15 px de alto** (restos finos de marco).
5. Se redimensiona **siempre a 60x80** para que la comparación con los moldes sea directa.

#### `reconocer_numero`: *template matching*

```python
_, vivo = cv2.threshold(molde_extraido, 127, 255, cv2.THRESH_BINARY)
vivo_pad = cv2.copyMakeBorder(vivo, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
resultado = cv2.matchTemplate(vivo_pad, molde_base, cv2.TM_CCOEFF_NORMED)
_, similitud, _, _ = cv2.minMaxLoc(resultado)
```

- Se añaden **10 px de margen negro** a la imagen en vivo antes de comparar. `matchTemplate` desliza el molde por encima, así compensa pequeños desplazamientos del número de 1 o 2 píxeles.
- `TM_CCOEFF_NORMED` devuelve la correlación normalizada (de -1 a 1). Es insensible a diferencias de brillo.
- Se compara con **los 10 moldes** y gana el de mayor similitud.
- **Umbral de seguridad 0.55:** si el mejor no llega, se devuelve `None` (probablemente es ruido, un trozo de dibujo o una carta mal leída) y esa lectura no entra en el historial.

**¿Por qué *template matching* y no OCR (Tesseract) ni una red neuronal?**

- Solo hay 10 clases posibles y siempre con la misma tipografía impresa.
- Es muy rápido y funciona sin entrenamiento: los moldes salen de la propia baraja.
- Como el número está normalizado a 60x80, la comparación es casi instantánea.
- Es completamente explicable, lo cual es importante en un proyecto académico.

### 3.7 Lógica del Mus (`logica_mus.py`)

La función `evaluar_mus(mano_cruda)` recibe una lista de enteros (los números leídos) y devuelve **cuatro cadenas** listas para pintar: `[pares, juego, grandes, chicas]`. Si hay menos de 4 cartas devuelve una lista vacía. Si hay más, usa las 4 primeras.

1. **Normalización (cerdos y pitos):** `12 si carta == 3`, `1 si carta == 2`. Todo el cálculo posterior se hace sobre esta mano normalizada.
2. **Pares:** `Counter(mano)` cuenta las repeticiones. Si hay un valor repetido 4 veces → *Duples (4 iguales)*; dos valores repetidos 2 veces → *Duples (2 pares)*; uno repetido 3 veces → *Medias*; uno repetido 2 veces → *Par*; en otro caso, no hay pareja.
3. **Juego:** suma con las figuras (10, 11, 12) valiendo 10. Si es ≥ 31 hay juego (y `(Max!)` si es exactamente 31); si no, se informa el punto.
4. **Grandes y chicas:** se ordena la mano de mayor a menor. Las dos primeras son las grandes y las dos últimas las chicas. Se muestran con nombre: `Rey/3`, `Caballo`, `Sota`, `Siete`, `Seis`, `Cinco`, `Cuatro`, `As/Pito`.

Esta capa **no usa OpenCV**: se puede probar pasándole una lista de números, por ejemplo `evaluar_mus([12, 3, 11, 1])`, que devuelve `['Pareja: Si | Par', 'Juego: Si | Puntos: 31 (Max!)', 'Grandes: Rey/3, Rey/3', 'Chicas: As/Pito, Caballo']` (el 3 cuenta como rey, así que hay dos reyes).

### 3.8 Configuración (`config_cartas.py`)

| Constante | Valor | Función |
|---|---|---|
| `DIRECTORIO_BASE`, `RUTA_MOLDES` | calculadas con `__file__` | Ruta a `moldes/`, válida desde cualquier carpeta de ejecución |
| `CAMARA_INDICE` | 2 | Índice de la cámara virtual (Iriun) |
| `CAMARA_ANCHO`, `CAMARA_ALTO` | 1920, 1080 | Resolución pedida |
| `CAMARA_BUFFER` | 1 | Anti-lag |
| `CARTA_ANCHO`, `CARTA_ALTO` | 400, 600 | Tamaño de la carta planchada |
| `AREA_MINIMA`, `AREA_MAXIMA` | 15000, 400000 | Filtro de tamaño de carta |
| `ASPECT_RATIO_MAX` | 2.5 | Proporción máxima lado largo/corto |
| `TRACKING_DISTANCIA_MAX` | 100 | Distancia máxima para "misma carta" |
| `TRACKING_VIDA_INICIAL` | 10 | Fotogramas de memoria sin ver la carta |
| `TRACKING_HISTORIAL_MAX` | 15 | Tamaño del historial de lecturas |

Los filtros en píxeles de `detector_cartas.py` (kernel de 45, ROI de 170, tamaños de dígito) están calibrados para la carta de **400x600**. Si cambias `CARTA_ANCHO` o `CARTA_ALTO`, hay que reescalarlos.

### 3.9 Decisiones de diseño y problemas resueltos

Cronología de los problemas más instructivos y cómo se resolvieron:

| Problema | Causa | Solución |
|---|---|---|
| El 5 de bastos no se leía | El número estaba pegado al marco, y el recorte incluía mesa oscura que se fusionaba con él | Borrar líneas con apertura morfológica (`quitar_lineas`) en lugar de filtrar por tamaño |
| Se leía un arco en lugar del 5 | La **esquina redondeada** de la carta pasaba el filtro de tamaño y era la mancha más cercana a la esquina | Descartar manchas con centro a menos de 35 px del borde |
| Algunas cartas seguían sin leerse | La esquina superior izquierda tenía restos de mesa | Probar también la esquina inferior derecha (carta girada 180°) |
| Restos de marco en los moldes | El margen de 4 px incluía una línea fina | Eliminar manchas de menos de 15 px de alto dentro del recorte |
| Cartas tumbadas deformadas | `ordenar_puntos` no sabe qué lado es largo | Parche anti-deformación con `np.roll` |
| Lecturas que cambian de un fotograma a otro | Ruido, sombras, reflejos | Memoria espacial con moda del historial |
| Imagen con retardo o la cámara por WiFi se colgaba | Cola de fotogramas y red | `CAP_PROP_BUFFERSIZE = 1` y cámara por cable USB (Iriun) |
| Una sola binarización no funciona siempre | Varía con la luz | Tres binarizaciones de respaldo |

### 3.10 Limitaciones

- Funciona mejor con **fondo oscuro y uniforme**. Un fondo claro reduce el contraste carta/mesa y degrada la detección de la silueta.
- La detección de la silueta usa un umbral adaptativo; si dos cartas se tocan, pueden unirse en un solo contorno y no pasar el filtro de proporción.
- Si la esquina de la carta queda tapada, no se puede leer.
- El umbral de 0.55 es empírico; con otra baraja o con otra cámara puede hacer falta recalibrarlo.
- La lógica evalúa **una sola mano** con 8 reyes. No compara manos de distintos jugadores ni ordena el ranking completo del juego (33, 34… 40, 32, 31).
- `ordenar_puntos` es ambiguo con cartas giradas cerca de 45°.

---

## 4. Parte B: detector de señas

### 4.1 Objetivo

Reconocer con la webcam las señas faciales típicas del Mus, tal como se definen en este proyecto:

| Seña | Gesto | Se detecta cuando… |
|---|---|---|
| **31** | Cerrar un ojo | Un ojo tiene pupila visible y el otro no |
| **Ciego** | Cerrar los dos ojos | Ninguno de los dos ojos tiene pupila visible |
| **Duples** | Levantar las cejas | El centro de gravedad de las cejas sube respecto a su altura de reposo |
| **Dos pitos** | Sacar la lengua | Una mancha rosada/roja grande aparece en la zona de la boca |

### 4.2 Archivos

| Archivo | Rol |
|---|---|
| `main_senas.py` | Abre la webcam, voltea la imagen, llama al detector y muestra el texto y las ventanas de depuración |
| `detector_senas.py` | Clase `DetectorSenasMus`: toda la visión y la máquina de estados |
| `config_senas.py` | Rutas de los modelos Haar, umbrales y rangos HSV |

### 4.3 Idea general

En lugar de buscar "un ojo cerrado" en toda la imagen (lo que sería muy difícil con técnicas clásicas), el sistema sigue una estrategia de **de lo general a lo particular**:

```
 frame ─▶ gris ─▶ CLAHE ─▶ detectar CARA (Haar) ─▶ suavizar la caja de la cara
                                                       │
                            ┌──────────────────────────┼──────────────────────────┐
                            ▼                          ▼                          ▼
                    ROI CEJAS (arriba)         ROI OJOS (centro)          ROI BOCA (abajo)
                    umbral + contornos         Haar de ojos + píxeles     HSV rojo + contorno
                    + centro de gravedad       oscuros de la pupila       más grande
                            │                          │                          │
                            └───────── contadores (máquina de estados) ───────────┘
                                                       │
                                              texto de la seña detectada
```

Una vez que se conoce la caja de la cara `(x, y, w, h)`, la posición de ojos, cejas y boca es **proporcional** a ella. Por eso las zonas de análisis (ROI) se calculan como fracciones del ancho y alto de la cara, y se adaptan a cualquier distancia a la cámara.

| Zona | Horizontal | Vertical | Color del recuadro |
|---|---|---|---|
| Cejas | 18 % – 82 % del ancho | 14 % – 30 % del alto | Morado |
| Ojos | 15 % – 85 % del ancho | 29 % – 53 % del alto | Azul |
| Boca | 25 % – 75 % del ancho | 65 % – 95 % del alto | Rojo |

### 4.4 `main_senas.py`

1. Abre la webcam `0` y crea `DetectorSenasMus`.
2. En cada fotograma lo voltea con `cv2.flip(frame, 1)` (**modo espejo**, para que tu ojo izquierdo aparezca a la izquierda).
3. Llama a `detector.procesar_fotograma(frame)`, que devuelve el fotograma con las cajas dibujadas, un diccionario de ventanas de depuración y la lista de textos de las señas activas.
4. Escribe los textos de las señas arriba a la izquierda, con un color por tipo (rojo para 31 y ciego, morado para duples, magenta para dos pitos).
5. Muestra las ventanas: gris, CLAHE, vídeo principal, y las de depuración de ojos, boca y cejas.

### 4.5 `DetectorSenasMus` paso a paso

#### Constructor `__init__`

- Carga los clasificadores **Haar** de cara y ojos desde las rutas de `config_senas.py`.
- Crea el ecualizador **CLAHE** (`clipLimit=2.0`, cuadrícula 8x8).
- Inicializa el historial de cajas de la cara, los **contadores** de cada seña y las **memorias** (altura de reposo de las cejas, y posición y tamaño habituales de los ojos).

#### `procesar_fotograma(frame)`

1. Convierte a gris y aplica **CLAHE**.

   *¿Por qué CLAHE?* La ecualización normal del histograma mejora el contraste global pero sobreexpone zonas. CLAHE (*Contrast Limited Adaptive Histogram Equalization*) lo hace por bloques de 8x8 con un límite, así que corrige la luz local de la cara (por ejemplo, un lado más sombreado) sin amplificar el ruido.
2. Detecta caras con Haar: `detectMultiScale(gray_clahe, scaleFactor=1.1, minNeighbors=5, minSize=(100,100))`. `minNeighbors=5` exige 5 detecciones vecinas para confirmar (menos falsos positivos) y `minSize=(100,100)` descarta caras diminutas.
3. **Media móvil** de la caja de la cara: guarda las últimas `FRAMES_MEMORIA` (5) cajas y usa su media. Sin esto, la caja temblaría de un fotograma a otro y con ella todas las zonas de análisis.
4. Dibuja la caja verde y llama a los tres análisis: cejas, ojos y lengua.

#### `analizar_ojos`: señas 31 y ciego

1. **Detección:** busca ojos con el Haar de ojos dentro de la ROI de los ojos (`minNeighbors=6`, tamaño mínimo 22 px y máximo el 22 % del ancho de la cara).
2. **Filtro anti-ceja:** el Haar a veces confunde una ceja con un ojo. Se separa la lista en candidatos izquierdo y derecho (según la mitad de la ROI donde caen) y, de cada lado, se queda con el más **bajo** (más lejos de la ceja).
3. **Memoria anatómica:** cuando se ven **los dos ojos** a la misma altura (diferencia < 5 % del alto de la cara), se guarda su altura y tamaño medios con una media exponencial (`0.8 · anterior + 0.2 · nuevo`). Después, un candidato se rechaza si queda **por encima** de la altura habitual o es demasiado grande (> 120 % del tamaño habitual): probablemente es una ceja.
4. **Medir si está abierto:** se recorta el **centro** del ojo (25–75 % del ancho, 25–65 % del alto), que es donde está la pupila, y se binariza con un **umbral dinámico** = 75 % de la luminosidad media del ojo. Se calcula el porcentaje de píxeles oscuros. Si supera `UMBRAL_PUPILA` (20 %), el ojo está **abierto**.

   *¿Por qué dinámico?* Un umbral fijo dependería de la luz de la habitación. Al ser relativo a la luminosidad del propio ojo, se adapta solo.
5. **Decisión:**

| Ojo izq. | Ojo der. | Efecto sobre los contadores |
|---|---|---|
| abierto | abierto | `contador_31` y `contador_ciego` bajan 3 |
| cerrado | cerrado | `contador_ciego` sube 2, `contador_31` baja 3 |
| uno abierto y otro cerrado | | `contador_31` sube 2, `contador_ciego` baja 2 |

6. Se muestra "Sena ciego detectada" o "Sena 31 detectada" cuando el contador alcanza `UMBRAL_FRAMES`. Si ambos lo alcanzan, tiene prioridad ciego.
7. Devuelve una imagen de depuración con la binarización de los dos ojos, lado a lado.

#### `analizar_cejas`: seña duples

Se detecta la **altura de las cejas relativa a su posición de reposo**.

1. Recorta la ROI de las cejas de la imagen con CLAHE y la suaviza (gaussiano 5x5).
2. **Binarización dinámica:** umbral = 78 % de la luz media de la zona, invertido, para que los pelos oscuros salgan blancos sobre la piel.
3. **Apertura 3x3** para separar pelos sueltos que toquen la ceja.
4. Busca contornos y **se queda solo con los dos mayores** (las dos cejas), que dibuja rellenos en un lienzo negro. Así se descartan arrugas y pelos.
5. Calcula el **centro de gravedad vertical** de los píxeles blancos, normalizado de 0 (techo de la zona) a 100 (suelo). Hace falta un mínimo de 30 píxeles blancos para fiarse.
6. **Línea base:** en el primer fotograma válido se guarda esa altura como la de reposo (`base_cejas_y`). La subida es `base − actual` (al subir las cejas, el valor baja, porque el eje Y de la imagen crece hacia abajo).
7. **Máquina de estados:** si la subida supera `UMBRAL_SUBIDA_CEJAS` (7.5 puntos), el contador sube 2. Si no, baja 2 y además la línea base se **actualiza lentamente** hacia la altura actual (`0.95 · base + 0.05 · actual`), lo que permite adaptarse a cambios de postura sin que se dispare la seña.
8. Muestra "Sena duples (cejas) detectada" cuando se alcanza el umbral de fotogramas.

#### `analizar_lengua`: seña dos pitos

1. Recorta la ROI de la boca de la imagen **a color** original.
2. Convierte a **HSV**. En HSV el color (H) está separado del brillo (V), así que el rojo/rosa de la lengua se detecta aunque cambie la luz. Con RGB, el mismo color cambia en los tres canales cuando varía la iluminación.
3. **Dos rangos** con `inRange` y se unen con OR. El rojo está en los dos extremos del círculo de tonos de HSV (H cerca de 0 y cerca de 179), así que hacen falta `BAJO/ALTO_ROJO_1` (H 0–12) y `BAJO/ALTO_ROJO_2` (H 165–179). Saturación mínima 110 y valor mínimo 70 para ignorar tonos apagados.
4. **Apertura 5x5:** borra las líneas finas (los labios cerrados son una línea rojiza de pocos píxeles de alto, la lengua es una mancha gruesa).
5. De todos los contornos, se queda **solo con el mayor** y calcula qué porcentaje de la ROI ocupa.
6. Si supera `UMBRAL_LENGUA` (8 %), el contador sube; si no, baja. Al alcanzar `UMBRAL_FRAMES`, se muestra "Sena dos pitos (lengua) detectada".

### 4.6 La máquina de estados temporal

Una seña **no se activa con un solo fotograma**. Cada seña tiene un contador que se comporta como un "cubo": sube cuando el gesto está presente y baja cuando no.

```
 contador ∈ [0 , MAX_CONTADOR = 14]       se muestra la seña si contador ≥ UMBRAL_FRAMES = 7
 gesto presente: +2          gesto ausente: −2  (o −3 en 31/ciego)
```

Con un incremento de 2 y un umbral de 7, el gesto debe mantenerse unos **4 fotogramas seguidos**. Esto elimina los falsos positivos de un parpadeo normal o de un fallo puntual del detector. El tope de 14 hace que el cartel desaparezca rápido (unos 4-5 fotogramas) cuando se deja de hacer el gesto, en lugar de tardar en vaciarse.

### 4.7 Configuración (`config_senas.py`)

| Constante | Valor | Función |
|---|---|---|
| `RUTA_CARA`, `RUTA_OJOS` | `../haarcascades/*.xml` | Modelos Haar (relativas a la carpeta de ejecución) |
| `FRAMES_MEMORIA` | 5 | Fotogramas de la media móvil de la cara |
| `UMBRAL_FRAMES` | 7 | Valor del contador a partir del cual se muestra la seña |
| `MAX_CONTADOR` | 14 | Tope de los contadores |
| `UMBRAL_PUPILA` | 20 | % de píxeles oscuros para ojo abierto |
| `UMBRAL_LENGUA` | 8 | % de boca ocupado por la lengua |
| `BAJO_ROJO_1` / `ALTO_ROJO_1` | `[0,110,70]` / `[12,255,255]` | Primer rango HSV del rojo |
| `BAJO_ROJO_2` / `ALTO_ROJO_2` | `[165,110,70]` / `[179,255,255]` | Segundo rango HSV del rojo |
| `UMBRAL_SUBIDA_CEJAS` | 7.5 | Puntos que deben subir las cejas |

### 4.8 Limitaciones

- **Un único usuario frente a la cámara.** El historial de la caja de la cara y las memorias (cejas, ojos) son únicos, no por persona.
- Necesita **cara de frente** y luz razonable. Las gafas, el pelo sobre las cejas o el contraluz empeoran los resultados.
- Si el Haar no detecta **ningún ojo** (por una mala iluminación o por perfil), el código lo interpreta como ojos cerrados y puede activar "ciego".
- La línea base de las cejas se toma del primer fotograma con detección válida: hay que empezar con las cejas relajadas. Si arranca con las cejas levantadas, la referencia será errónea hasta que se adapte lentamente.
- La detección de la lengua se basa en el color: maquillaje rojo o un fondo rojizo dentro de la ROI pueden dar falsos positivos.
- Los umbrales de este proyecto se han calibrado con unas condiciones concretas; con otra persona o iluminación puede ser necesario ajustarlos.

---

## 5. Cómo extender el proyecto

| Quiero… | Dónde tocar |
|---|---|
| Soportar otras reglas (4 reyes) | `logica_mus.py`: cambiar la normalización (en 4 reyes el 3 y el 2 son cartas normales) |
| Mostrar el ranking completo del juego (31 > 32 > 40 > 37 …) | `logica_mus.py`: añadir una tabla de orden y devolver la posición |
| Reconocer palos | Nuevo módulo en `detector_cartas.py` (por ejemplo, por color dominante + forma) y moldes de palos |
| Evaluar las manos de varios jugadores | `main_cartas.py`: agrupar las cartas por zonas de la imagen y llamar a `evaluar_mus` por grupo |
| Añadir una seña nueva | `detector_senas.py`: método `analizar_*` nuevo, un contador en `__init__`, y constantes en `config_senas.py` |
| Cambiar de cámara o resolución | `config_cartas.py` (`CAMARA_*`) o `cv2.VideoCapture(...)` en `main_senas.py` |
| Usar otra baraja | Regenerar los moldes con `generador_moldes.py` y recalibrar el umbral de 0.55 |

---

## 6. Glosario

| Término | Significado |
|---|---|
| **Binarizar / umbralizar** | Convertir una imagen en gris a blanco y negro puro según un umbral |
| **Otsu** | Método que calcula automáticamente el mejor umbral global |
| **Umbral adaptativo** | Umbral calculado por zonas, no uno global |
| **Morfología** | Operaciones (apertura, cierre, dilatación…) sobre las formas de una imagen binaria |
| **Apertura** | Erosión seguida de dilatación: elimina estructuras más pequeñas que el kernel |
| **Cierre** | Dilatación seguida de erosión: rellena huecos |
| **Kernel** | Pequeña matriz que define el tamaño y forma de una operación morfológica o un filtro |
| **Contorno** | Línea que rodea una mancha blanca de una imagen binaria |
| **ROI** | *Region Of Interest*: zona de la imagen que se analiza |
| **Transformación de perspectiva** | Proyección que convierte un cuadrilátero visto en perspectiva en un rectángulo recto |
| **Template matching** | Búsqueda de dónde y cuánto se parece una imagen pequeña (molde) a otra |
| **Molde** | Imagen de referencia de 60x80 de un número, usada como plantilla |
| **Haar cascade** | Clasificador clásico (Viola-Jones) que detecta objetos con patrones de claros y oscuros |
| **CLAHE** | Ecualización adaptativa del contraste por bloques, con límite |
| **HSV** | Espacio de color: Tono (Hue), Saturación y Valor (brillo) |
| **Moda** | El valor que más veces aparece en una lista |
| **HUD** | *Heads-Up Display*: la información dibujada sobre el vídeo |
| **Tracking** | Seguir el mismo objeto a lo largo de varios fotogramas |
| **Cerdos y pitos** | En 8 reyes, los 3 (cerdos) valen como reyes y los 2 (pitos) como ases |
