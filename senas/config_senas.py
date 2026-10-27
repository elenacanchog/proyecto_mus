# ==============================================================================
# config_senas.py
# Archivo de configuración central: Umbrales, rutas y variables de calibración
# ==============================================================================
import numpy as np

# 1. Rutas de los modelos Haar Cascade preentrenados de Viola-Jones
RUTA_CARA = '../haarcascades/haarcascade_frontalface_default.xml'
RUTA_OJOS = '../haarcascades/haarcascade_eye.xml'

# 2. Máquina de estados temporal (Filtros anti-parpadeo y estabilización)
FRAMES_MEMORIA = 5             # Cantidad de fotogramas a recordar para estabilizar la caja de la cara
UMBRAL_FRAMES = 7              # Fotogramas seguidos necesarios para confirmar una seña real
MAX_CONTADOR = 14              # Tope máximo del contador (para que el cartel no tarde en borrarse)

# 3. Variables de calibración para los OJOS (Seña 31 y Ciego)
UMBRAL_PUPILA = 20             # Porcentaje de píxeles oscuros necesarios para considerar el ojo ABIERTO

# 4. Variables de calibración para la BOCA (Seña Dos Pitos / Lengua)
UMBRAL_LENGUA = 8              # Porcentaje del recuadro de la boca ocupado por la lengua
# Rangos en espacio HSV (Hue, Saturation, Value) para aislar tonos rosados/rojizos
BAJO_ROJO_1 = np.array([0, 110, 70])
ALTO_ROJO_1 = np.array([12, 255, 255])
BAJO_ROJO_2 = np.array([165, 110, 70])
ALTO_ROJO_2 = np.array([179, 255, 255])

# 5. Variables de calibración para las CEJAS (Seña Duples)
UMBRAL_SUBIDA_CEJAS = 7.5      # Puntos porcentuales que deben subir las cejas respecto al reposo