# Proyecto Mus: visión artificial para el juego del Mus

Proyecto de **visión artificial con Python y OpenCV** que ayuda a jugar al Mus. Tiene dos programas independientes:

| Programa | Qué hace |
|---|---|
| **Detector de cartas** (`cartas/`) | Lee con la cámara las 4 cartas de una mano de la baraja española, reconoce el número de cada una y dice qué jugadas hay: pares, juego, grandes y chicas. |
| **Detector de señas** (`senas/`) | Detecta con la webcam las señas faciales típicas del Mus (31, ciego, duples y dos pitos). |

Todo funciona **sin redes neuronales ni modelos entrenados por nosotros**: usa técnicas clásicas de visión (umbralizado, morfología, contornos, transformaciones de perspectiva, *template matching*, clasificadores Haar y espacio de color HSV). La explicación detallada de cada técnica está en [`DOCUMENTACION.md`](DOCUMENTACION.md).

---

## Índice

1. [Qué hace cada funcionalidad](#1-qué-hace-cada-funcionalidad)
2. [Estructura del proyecto](#2-estructura-del-proyecto)
3. [Requisitos](#3-requisitos)
4. [Instalación paso a paso](#4-instalación-paso-a-paso)
5. [Configurar la cámara](#5-configurar-la-cámara)
6. [Cómo ejecutar](#6-cómo-ejecutar)
7. [Consejos para que funcione bien](#7-consejos-para-que-funcione-bien)
8. [Calibración](#8-calibración)
9. [Solución de problemas](#9-solución-de-problemas)
10. [Limitaciones conocidas](#10-limitaciones-conocidas)

---

## 1. Qué hace cada funcionalidad

### 1.1 Detector de cartas

Colocas **4 cartas** sobre la mesa, la cámara las ve, y el programa dibuja sobre el vídeo:

- Un recuadro verde alrededor de cada carta y su **número** (1, 2, 3, 4, 5, 6, 7, 10, 11 o 12; los palos no se reconocen).
- Las jugadas de la mano, con las reglas de **8 reyes** (el 3 vale como un rey y el 2 como un as):

```
Mesa: [12, 3, 11, 1]
Pareja: Si | Par
Juego: Si | Puntos: 31 (Max!)
Grandes: Rey/3, Rey/3
Chicas: As/Pito, Caballo
```

Para reconocer los números se usan unos **moldes** (una imagen por número) que se generan una sola vez a partir de una foto de la baraja. Después, cada carta se compara en directo con esos moldes.

### 1.2 Detector de señas

Con la webcam, el programa localiza tu cara y analiza tres zonas de ella:

| Seña | Gesto | Cómo se detecta |
|---|---|---|
| **31** | Cerrar un solo ojo | Un ojo abierto y otro cerrado |
| **Ciego** | Cerrar los dos ojos | Los dos ojos cerrados |
| **Duples** | Levantar las cejas | Las cejas suben respecto a su posición en reposo |
| **Dos pitos** | Sacar la lengua | Aparece una mancha de color rosado/rojo en la boca |

El texto de la seña aparece arriba a la izquierda cuando el gesto se mantiene unos fotogramas seguidos (así los parpadeos normales no cuentan).

---

## 2. Estructura del proyecto

```
proyecto_mus/
├── README.md                  ← este archivo
├── DOCUMENTACION.md           ← explicación técnica de los dos programas
├── requirements.txt           ← dependencias de Python
├── baraja.jpg                 ← foto de la baraja (solo para generar los moldes)
├── haarcascades/              ← modelos Haar de OpenCV (cara y ojos)
│   ├── haarcascade_frontalface_default.xml
│   └── haarcascade_eye.xml
│
├── cartas/                    ← PROGRAMA 1: detector de cartas y jugadas del Mus
│   ├── main_cartas.py         ← ★ EJECUTABLE principal
│   ├── config_cartas.py       ← parámetros ajustables (cámara, tamaños, tracking)
│   ├── detector_cartas.py     ← motor de visión (extraer y reconocer el número)
│   ├── logica_mus.py          ← reglas del Mus (pares, juego, grandes, chicas)
│   ├── moldes/                ← moldes definitivos: molde_1.png … molde_12.png
│   ├── generador_moldes.py    ← herramienta: genera los moldes desde baraja.jpg
│   ├── moldes_generados/      ← salida de generador_moldes.py (se sobrescribe)
│   ├── conexion_camara.py     ← prueba: comprobar cámara y FPS
│   └── mus_directo.py         ← versión monolítica previa (referencia histórica)
│
└── senas/                     ← PROGRAMA 2: detector de señas faciales
    ├── main_senas.py          ← ★ EJECUTABLE principal
    ├── config_senas.py        ← umbrales y rangos de color
    └── detector_senas.py      ← clase DetectorSenasMus (toda la visión)
```

**Idea de diseño:** cada programa se divide en tres capas con una única responsabilidad:

| Capa | Archivo | Responsabilidad |
|---|---|---|
| Configuración | `config_*.py` | Constantes ajustables. No contiene lógica. |
| Motor | `detector_*.py` / `logica_mus.py` | Procesa imágenes o aplica las reglas. No abre la cámara ni dibuja ventanas. |
| Ejecución | `main_*.py` | Abre la cámara, une las piezas y dibuja el resultado. |

### Archivos ejecutables

| Archivo | Tipo | Para qué sirve |
|---|---|---|
| `cartas/main_cartas.py` | **Principal** | Detector de cartas y jugadas en directo |
| `senas/main_senas.py` | **Principal** | Detector de señas en directo |
| `cartas/generador_moldes.py` | Herramienta | Crea los moldes de los números a partir de `baraja.jpg` |
| `cartas/conexion_camara.py` | Prueba | Comprueba que la cámara funciona y mide los FPS |
| `cartas/mus_directo.py` | Referencia | Versión anterior, todo en un solo archivo |

---

## 3. Requisitos

- **Sistema operativo:** Ubuntu (probado). Debería funcionar también en otros Linux, macOS y Windows.
- **Python 3.8 o superior.**
- **Paquetes de Python:** `opencv-python` y `numpy` (se instalan con `requirements.txt`).
- **Una cámara:**
  - Para las **señas**: cualquier webcam (el programa usa la cámara `0`).
  - Para las **cartas**: recomendada la cámara del móvil conectada por USB con [Iriun Webcam](https://iriun.com/), mejor que la del portátil porque se puede colocar mirando la mesa desde arriba. Sirve cualquier webcam.
- **Entorno gráfico:** el programa abre ventanas con `cv2.imshow`.

---

## 4. Instalación paso a paso

### 4.1 Clonar el repositorio

```bash
git clone https://github.com/elenacanchog/proyecto_mus.git
cd proyecto_mus
```

### 4.2 Crear y activar el entorno virtual

En Ubuntu puede que antes necesites el soporte para entornos virtuales:

```bash
sudo apt update
sudo apt install python3-venv python3-pip
```

Crea el entorno (queda en una carpeta `venv_mus`) y actívalo:

```bash
python3 -m venv venv_mus
source venv_mus/bin/activate
```

Sabrás que está activo porque el terminal muestra `(venv_mus)` al principio de la línea. Para salir de él más adelante: `deactivate`.

> En Windows la activación es `venv_mus\Scripts\activate`.

### 4.3 Instalar las dependencias

Con el entorno activo:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Comprueba que OpenCV se instaló bien:

```bash
python -c "import cv2; print(cv2.__version__)"
```

> Instala `opencv-python` (el que trae `requirements.txt`) y **no** `opencv-python-headless`: la versión *headless* no puede abrir ventanas.

### 4.4 Modelos Haar (solo para las señas)

El programa de señas necesita dos archivos en la carpeta `haarcascades/` de la raíz del proyecto. Si no están en el repositorio, cópialos desde tu instalación de OpenCV:

```bash
mkdir -p haarcascades
RUTA=$(python -c "import cv2; print(cv2.data.haarcascades)")
cp "$RUTA/haarcascade_frontalface_default.xml" haarcascades/
cp "$RUTA/haarcascade_eye.xml" haarcascades/
```

### 4.5 Comprobar los moldes (solo para las cartas)

La carpeta `cartas/moldes/` debe contener 10 archivos: `molde_1.png`, `molde_2.png`, `molde_3.png`, `molde_4.png`, `molde_5.png`, `molde_6.png`, `molde_7.png`, `molde_10.png`, `molde_11.png` y `molde_12.png`. Vienen incluidos en el repositorio. Si faltan, mira [cómo regenerarlos](#generar-o-regenerar-los-moldes).

---

## 5. Configurar la cámara

### Detector de cartas

1. Instala **Iriun Webcam** en el móvil y su cliente en el ordenador, y conecta el móvil por USB (o por WiFi, con algo más de retardo).
2. Averigua qué índice tiene la cámara virtual:

   ```bash
   sudo apt install v4l-utils
   v4l2-ctl --list-devices
   ```

   O prueba índices con Python:

   ```bash
   python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(6)]"
   ```

3. Escribe ese número en `cartas/config_cartas.py`:

   ```python
   CAMARA_INDICE = 2      # ← cambia este valor
   ```

4. Comprueba la imagen y los FPS con la prueba de cámara (edita también el `indice_camara` dentro de `conexion_camara.py`):

   ```bash
   cd cartas
   python conexion_camara.py
   ```

### Detector de señas

Usa la cámara `0` (la webcam del portátil). Si necesitas otra, cambia `cv2.VideoCapture(0)` en `senas/main_senas.py`.

---

## 6. Cómo ejecutar

Activa siempre el entorno virtual primero (`source venv_mus/bin/activate`). Pulsa **`Q`** en la ventana de vídeo para salir.

> **Importante:** ejecuta cada programa **desde su propia carpeta** (`cd cartas` o `cd senas`). Algunas rutas (`../haarcascades`, `../baraja.jpg`) son relativas a la carpeta desde la que lanzas el comando.

### Detector de cartas

```bash
cd cartas
python main_cartas.py
```

Verás la imagen de la cámara. Pon 4 cartas sobre la mesa y el programa mostrará el número de cada una y las jugadas. Mientras haya menos de 4 cartas leídas muestra `Esperando 4 cartas...`.

### Detector de señas

```bash
cd senas
python main_senas.py
```

Se abren varias ventanas: la imagen con tu cara y las zonas analizadas, y ventanas de depuración con los filtros (escala de grises, CLAHE, ojos, boca y cejas). Hay que mirar de frente a la cámara, con la cara bien iluminada. Al arrancar, mantén las cejas relajadas un momento: ese instante sirve de referencia para detectar el gesto de levantarlas.

### Generar o regenerar los moldes

Solo hace falta si quieres crear los moldes de nuevo (por ejemplo, con otra baraja). Necesitas una foto `baraja.jpg` en la raíz del proyecto con las cartas 1, 2, 3, 4, 5, 6, 7, 10, 11 y 12 en fila, cada una separada de las demás y vista de frente.

```bash
cd cartas
python generador_moldes.py
```

El programa muestra el número leído de cada carta y guarda los resultados en `cartas/moldes_generados/` con nombres como `carta_generada_01.png`. Después:

1. Abre cada imagen y comprueba que el número está limpio y completo.
2. Renómbrala con el valor real de la carta y cópiala a `cartas/moldes/`, por ejemplo `molde_5.png`.

Los archivos de `moldes_generados/` se sobrescriben en cada ejecución. Por eso los definitivos viven en `moldes/`.

### Pruebas

```bash
cd cartas
python conexion_camara.py     # comprueba la cámara y muestra los FPS
python mus_directo.py         # versión antigua, todo en un archivo (solo referencia)
```

---

## 7. Consejos para que funcione bien

**Para las cartas:**

- Usa un **paño o cartulina negra mate** como mesa. Con un fondo oscuro, las cartas (claras) se distinguen mucho mejor. Con mesas claras la detección se degrada.
- Pon la cámara **fija y mirando desde arriba**, lo más perpendicular posible a la mesa.
- Acércala hasta que cada carta ocupe un buen tamaño en la imagen: los números deben verse nítidos.
- Luz **uniforme y difusa**, sin reflejos sobre las cartas.
- Deja **espacio entre las cartas** para que no se toquen ni se solapen.
- Las cartas pueden ir giradas o tumbadas; el programa las endereza.

**Para las señas:**

- Cara de frente, bien iluminada y sin contraluz.
- Las gafas pueden interferir con la detección de los ojos.
- Si cambias de persona o de luz, puede que haga falta recalibrar los umbrales (ver abajo).

---

## 8. Calibración

Todos los parámetros ajustables están en los archivos de configuración, con comentarios.

**`cartas/config_cartas.py`**

| Parámetro | Qué controla | Cuándo tocarlo |
|---|---|---|
| `CAMARA_INDICE` | Índice de la cámara | Si no abre o abre otra cámara |
| `CAMARA_ANCHO` / `CAMARA_ALTO` | Resolución pedida a la cámara | Si va lento, baja a 1280x720 |
| `AREA_MINIMA` / `AREA_MAXIMA` | Área (px) para considerar algo una carta | Si detecta ruido, sube la mínima; si alejas la cámara, bájala |
| `ASPECT_RATIO_MAX` | Proporción máxima lado largo/corto | Si detecta objetos alargados que no son cartas |
| `TRACKING_DISTANCIA_MAX` | Distancia para considerar que es la misma carta entre fotogramas | Si las cartas se mueven mucho |
| `TRACKING_VIDA_INICIAL` | Fotogramas que se recuerda una carta no vista | Si parpadean los recuadros |
| `TRACKING_HISTORIAL_MAX` | Lecturas usadas para decidir el número estable | Más alto = más estable pero más lento en reaccionar |

El umbral de reconocimiento (55 % de similitud) está en `detector_cartas.py`, dentro de `reconocer_numero`.

**`senas/config_senas.py`**

| Parámetro | Qué controla |
|---|---|
| `UMBRAL_PUPILA` | % de píxeles oscuros para considerar un ojo abierto |
| `UMBRAL_LENGUA` | % de la boca que debe ser rosado/rojo para detectar la lengua |
| `BAJO_ROJO_*` / `ALTO_ROJO_*` | Rangos HSV del color de la lengua |
| `UMBRAL_SUBIDA_CEJAS` | Cuánto deben subir las cejas para contar como seña |
| `FRAMES_MEMORIA` | Fotogramas usados para suavizar el recuadro de la cara |
| `UMBRAL_FRAMES` / `MAX_CONTADOR` | Fotogramas seguidos para confirmar una seña / tope del contador |

---

## 9. Solución de problemas

| Problema | Causa probable y solución |
|---|---|
| `Error: No se pudo abrir la cámara` | Índice equivocado. Prueba con los índices 0, 1, 2, 3 (ver [sección 5](#5-configurar-la-cámara)). |
| `ERROR CRÍTICO: No existe la carpeta de moldes` | Falta `cartas/moldes/` o está vacía. Ver [4.5](#45-comprobar-los-moldes-solo-para-las-cartas). |
| Error al cargar la cámara en Iriun | Abre primero la app del móvil y el cliente del PC, y después el programa. |
| El programa de señas no detecta la cara | No se encuentran los `.xml` de `haarcascades/`, o lo ejecutas fuera de la carpeta `senas/`. Ver [4.4](#44-modelos-haar-solo-para-las-señas). |
| `cv2.error ... The function is not implemented` al abrir ventanas | Tienes `opencv-python-headless`. Ejecuta `pip uninstall opencv-python-headless` y `pip install opencv-python`. |
| No detecta las cartas o dibuja el recuadro mal | Mesa demasiado clara, poca luz o cartas muy lejos. Usa paño negro y acerca la cámara; ajusta `AREA_MINIMA`. |
| Lee mal algunos números | Imagen borrosa o con reflejos. Mejora luz y enfoque; baja el umbral de 0.55 con cuidado (puede dar falsos positivos). |
| La imagen va con retraso | Usa el cable USB en lugar de WiFi, o baja la resolución en `config_cartas.py`. |
| Las señas se activan solas o no se activan | Recalibra `UMBRAL_PUPILA`, `UMBRAL_LENGUA` y `UMBRAL_SUBIDA_CEJAS` con tu iluminación. |

---

## 10. Limitaciones conocidas

- Solo se leen los **números**; los palos no se reconocen (no hacen falta para estas jugadas).
- Las jugadas se evalúan para **una sola mano de 4 cartas** y con las reglas de **8 reyes**. No se comparan manos entre jugadores ni se ordena el ranking completo de juego (solo se marca el 31 como la mejor puntuación).
- La detección de cartas funciona mejor con **fondo oscuro y uniforme**.
- Una carta cuya esquina quede tapada (por una mano o por otra carta) no se puede leer.
- Las señas dependen mucho de la **iluminación** y de que la cara esté de frente. Si el detector no localiza ningún ojo, el programa lo interpreta como ojos cerrados.

---

## Documentación técnica

Cómo funciona cada programa por dentro, qué técnicas se usan, por qué y qué hace cada función: **[DOCUMENTACION.md](DOCUMENTACION.md)**.
