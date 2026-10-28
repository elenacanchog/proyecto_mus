"""
======================================================================
MOTOR DE VISIÓN ARTIFICIAL (DETECTOR DE CARTAS)
======================================================================
Este archivo contiene exclusivamente la lógica de procesamiento de imagen.
Se encarga de cargar los moldes, limpiar las imágenes de la cámara, 
extraer los dígitos de las esquinas y compararlos (OCR).
No sabe nada de las reglas del Mus ni de la cámara, solo analiza píxeles.
"""
import cv2
import numpy as np
import os

# Importamos la ruta de los moldes desde nuestro archivo de configuración
from config_cartas import RUTA_MOLDES

# ==============================================================================
# 1. CARGA DEL DICCIONARIO DE MOLDES (El cerebro del OCR)
# ==============================================================================
def cargar_diccionario_moldes():
    """
    Busca en la carpeta de moldes las imágenes (molde_1.png, molde_2.png...)
    y las carga en memoria RAM binarizadas para poder compararlas a máxima velocidad.
    """
    moldes = {}
    
    if not os.path.exists(RUTA_MOLDES):
        print(f"ERROR CRÍTICO: No existe la carpeta de moldes en {RUTA_MOLDES}.")
        print("Asegúrate de que la carpeta 'moldes' está en el mismo directorio.")
        exit()

    for archivo in os.listdir(RUTA_MOLDES):
        # Filtramos para asegurarnos de que solo leemos los archivos correctos
        if archivo.startswith("molde_") and archivo.endswith(".png"):
            # Extraemos el número del nombre del archivo (ej: de "molde_12.png" sacamos "12")
            numero_str = archivo.replace("molde_", "").replace(".png", "")
            
            try:
                numero_real = int(numero_str)
                ruta = os.path.join(RUTA_MOLDES, archivo)
                molde_img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                
                if molde_img is not None:
                    # Binarizamos (blanco y negro puro) para que la comparación matemática sea exacta
                    _, molde_bin = cv2.threshold(molde_img, 127, 255, cv2.THRESH_BINARY)
                    moldes[numero_real] = molde_bin
            except ValueError:
                pass # Ignoramos archivos basura que no tengan un número válido
                
    print(f"[OK] Motor de Visión: Diccionario cargado con {len(moldes)} moldes perfectos.")
    return moldes

def reconocer_numero(molde_extraido, diccionario_moldes):
    """
    Compara la mancha extraída de la cámara con todos nuestros moldes perfectos.
    Usa el algoritmo 'matchTemplate' con un margen de deslizamiento para 
    absorber los temblores de la cámara. Especialmente útil para números dobles (11, 12).
    """
    mejor_coincidencia = None
    max_similitud = -1

    # Aseguramos que la imagen extraída de la cámara es blanco/negro puro (quitamos grises/ruido)
    _, molde_vivo_bin = cv2.threshold(molde_extraido, 127, 255, cv2.THRESH_BINARY)
    
    # EL TRUCO DEL DESLIZAMIENTO:
    # Añadimos 10 píxeles de margen negro alrededor de la imagen capturada. 
    # Esto permite que el algoritmo deslice nuestro molde por encima de la imagen en vivo 
    # hasta que encaje como un puzzle, compensando si la cámara tembló 1 o 2 milímetros.
    molde_vivo_padded = cv2.copyMakeBorder(molde_vivo_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    for numero_real, molde_base in diccionario_moldes.items():
        # TM_CCOEFF_NORMED es el mejor algoritmo para emparejar patrones con ruido
        resultado = cv2.matchTemplate(molde_vivo_padded, molde_base, cv2.TM_CCOEFF_NORMED)
        
        # minMaxLoc nos dice el punto exacto donde más se parecen y su porcentaje de similitud
        _, similitud, _, _ = cv2.minMaxLoc(resultado)
        
        if similitud > max_similitud:
            max_similitud = similitud
            mejor_coincidencia = numero_real

    # Umbral de seguridad: si supera el 55% de similitud estructural, aceptamos el match.
    # Si es menor, es probable que esté viendo un trozo de dibujo o ruido.
    if max_similitud > 0.55:
        return mejor_coincidencia
    return None

# ==============================================================================
# 2. FUNCIONES MATEMÁTICAS DE GEOMETRÍA Y EXTRACCIÓN
# ==============================================================================
def ordenar_puntos(puntos):
    """
    Fuerza que las 4 esquinas de la carta tengan siempre este orden estricto:
    1: Arriba-Izquierda, 2: Arriba-Derecha, 3: Abajo-Derecha, 4: Abajo-Izquierda.
    Vital para que al 'planchar' la carta no quede girada ni espejada.
    """
    puntos_ordenados = np.zeros((4, 2), dtype="float32")
    # Sumando x+y, el punto más pequeño es la esquina superior izquierda y el mayor la inferior derecha
    suma = puntos.sum(axis=1)
    puntos_ordenados[0] = puntos[np.argmin(suma)]
    puntos_ordenados[2] = puntos[np.argmax(suma)]
    
    # Restando y-x, el menor es la superior derecha y el mayor la inferior izquierda
    resta = np.diff(puntos, axis=1)
    puntos_ordenados[1] = puntos[np.argmin(resta)]
    puntos_ordenados[3] = puntos[np.argmax(resta)]
    return puntos_ordenados

def quitar_lineas(binaria):
    """
    Borra las líneas rectas horizontales y verticales (como el marco negro de la carta).
    Al usar kernels de 45 píxeles de largo, las líneas mueren, pero el número (que es curvo y corto) sobrevive.
    """
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    k_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    
    vert = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_vert)
    hor = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_hor)
    
    lineas = cv2.bitwise_or(vert, hor)
    lineas = cv2.dilate(lineas, np.ones((3, 3), np.uint8))
    
    # Restamos las líneas a la imagen original
    return cv2.bitwise_and(binaria, cv2.bitwise_not(lineas))

def buscar_digito(binaria_original):
    """
    Analiza una esquina de 170x170 de la carta limpia de marcos.
    Encuentra las manchas de tinta, filtra la basura (polvo, dibujos centrales) 
    y envuelve el número en una caja de 60x80 píxeles.
    """
    limpia = quitar_lineas(binaria_original)
    
    # Busca todas las manchas de tinta desconectadas
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(limpia, connectivity=8)
    alto_roi, ancho_roi = limpia.shape
    candidatos = []
    
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        
        # 1. Filtro de Tamaño: Descartes de polvo (muy pequeño) o dibujos grandes
        if not (4 < w < 60 and 18 < h < 60 and area > 60): 
            continue
            
        # 2. Filtro de Bordes: Si toca la pared del recorte, está cortado, lo ignoramos
        if x <= 0 or y <= 0 or x + w >= ancho_roi or y + h >= alto_roi: 
            continue
        
        # 3. FILTRO MAESTRO DEL 5 DE BASTOS: Descartar esquinas redondeadas.
        # Si el centroide de la mancha está demasiado pegado a la esquina superior izquierda (< 35px),
        # es el arco de la carta o un trozo de mesa, no el número.
        centro_x = x + w / 2
        centro_y = y + h / 2
        if centro_x < 35 or centro_y < 35: 
            continue

        candidatos.append((x, y, w, h))

    # Si no hay nada que parezca un número, abortamos
    if not candidatos: 
        return None, None

    # ANCLAJE: Ordenamos de arriba-izquierda a abajo-derecha. 
    # El candidato más arriba a la izquierda es el primer dígito (ej: el '1' del '12').
    candidatos.sort(key=lambda f: f[0] + f[1])
    x0, y0, w0, h0 = candidatos[0]
    cy0 = y0 + h0 / 2
    
    # AGRUPACIÓN: Si hay otro dígito cerca en el eje X y a la misma altura en Y, lo sumamos (para el 10, 11, 12).
    grupo = [f for f in candidatos if abs(f[0] - x0) < 50 and abs((f[1] + f[3] / 2) - cy0) < 15]

    # Calculamos la caja final que envuelve a todo el grupo
    x_min = min(f[0] for f in grupo)
    y_min = min(f[1] for f in grupo)
    x_max = max(f[0] + f[2] for f in grupo)
    y_max = max(f[1] + f[3] for f in grupo)

    margen = 4
    x1, y1 = max(0, x_min - margen), max(0, y_min - margen)
    x2, y2 = min(ancho_roi, x_max + margen), min(alto_roi, y_max + margen)
    
    # Recortamos la imagen ya limpia de líneas y basuras
    recorte = limpia[y1:y2, x1:x2]

    # Limpieza final: Si quedó alguna mota de polvo dentro del recorte, la borramos
    n2, et2, st2, _ = cv2.connectedComponentsWithStats(recorte, connectivity=8)
    solo_digitos = np.zeros_like(recorte)
    for j in range(1, n2):
        if st2[j, cv2.CC_STAT_HEIGHT] >= 15:
            solo_digitos[et2 == j] = 255

    # Estandarizamos SIEMPRE a 60x80 píxeles para que matchTemplate funcione a la perfección
    return cv2.resize(solo_digitos, (60, 80)), (x1, y1, x2, y2)

def extraer_numero(carta):
    """
    Función principal de extracción. Recibe la carta planchada en color.
    Prueba varios niveles de iluminación para asegurar que extrae el número
    incluso si hay sombras en la mesa, probando tanto la esquina superior izquierda
    como la inferior derecha (carta girada 180º).
    """
    for girar in (False, True):
        # Si no lo encontramos arriba a la izquierda, giramos la carta y miramos la otra esquina
        c = cv2.rotate(carta, cv2.ROTATE_180) if girar else carta
        
        # Zona segura (170x170): Suficientemente grande para coger el número, 
        # pero evitando los dibujos del centro de la carta.
        roi = c[0:170, 0:170]
        gris_suavizado = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)

        # 3 Estrategias de binarización distintas (una para mucha luz, otra para sombras, otra intermedia)
        _, b1 = cv2.threshold(gris_suavizado, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b2 = cv2.adaptiveThreshold(gris_suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        _, b3 = cv2.threshold(gris_suavizado, 100, 255, cv2.THRESH_BINARY_INV)

        # Probamos los 3 filtros. El primero que devuelva un número válido, gana.
        for b in (b1, b2, b3):
            molde, bbox = buscar_digito(b)
            if molde is not None:
                return molde, bbox
                
    return None, None