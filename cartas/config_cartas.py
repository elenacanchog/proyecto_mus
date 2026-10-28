"""
======================================================================
ARCHIVO DE CONFIGURACIÓN
======================================================================
Aquí guardamos todas las constantes y parámetros ajustables del sistema.
Si necesitas calibrar la cámara o la detección, hazlo aquí.
"""
import os

# --- RUTAS DE DIRECTORIOS ---
# Buscamos la carpeta 'moldes' en el mismo directorio donde se ejecute el programa
DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_MOLDES = os.path.join(DIRECTORIO_BASE, 'moldes')

# --- CONFIGURACIÓN DE LA CÁMARA (IRIUN) ---
# Índice de la cámara virtual en Ubuntu (suele ser 0, 1 o 2)
CAMARA_INDICE = 2
CAMARA_ANCHO = 1920
CAMARA_ALTO = 1080
# Buffer a 1 obliga a OpenCV a procesar siempre el frame en tiempo real (mata el lag)
CAMARA_BUFFER = 1

# --- PARÁMETROS DE VISIÓN Y GEOMETRÍA ---
# Tamaño al que se "plancha" cada carta matemáticamente
CARTA_ANCHO = 400
CARTA_ALTO = 600

# Filtros de tamaño para detectar cartas en la mesa (en píxeles de área)
# Si el programa detecta ruido de la mesa, sube el AREA_MINIMA.
# Si alejas la cámara, baja el AREA_MINIMA.
AREA_MINIMA = 15000
AREA_MAXIMA = 400000

# Proporción máxima permitida para considerar que un contorno es una carta
# (Evita que el sistema confunda un lápiz o un brazo con una carta)
ASPECT_RATIO_MAX = 2.5 

# --- MEMORIA Y ESTABILIDAD (TRACKING) ---
# Distancia máxima (en píxeles) para considerar que una carta es la misma en el siguiente frame
TRACKING_DISTANCIA_MAX = 100
# Cuántos frames aguanta una carta en la memoria si la cámara la pierde un instante (sombras)
TRACKING_VIDA_INICIAL = 10
# Cuántos resultados guardamos en el historial para calcular el número más estable (la moda)
TRACKING_HISTORIAL_MAX = 15