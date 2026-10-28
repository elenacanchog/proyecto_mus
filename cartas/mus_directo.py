import cv2
import numpy as np
import os
import time
import math                     # NUEVO: Para calcular distancias entre cartas
from collections import Counter # NUEVO: Para sacar el número más repetido (la moda)

# ==============================================================================
# 1. CARGA DEL DICCIONARIO DE MOLDES (El cerebro del OCR)
# ==============================================================================
def cargar_diccionario_moldes():
    """
    Busca en la carpeta 'moldes' las imágenes molde_1.png, molde_2.png, etc.
    Las carga en memoria binarizadas para compararlas súper rápido en directo.
    """
    # Construimos la ruta relativa a este script (busca la carpeta 'moldes' al lado de mus_directo.py)
    carpeta = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'moldes')
    moldes = {}
    
    if not os.path.exists(carpeta):
        print(f"ERROR: No existe la carpeta {carpeta}.")
        exit()

    for archivo in os.listdir(carpeta):
        # Filtramos para coger solo los png que empiecen por "molde_"
        if archivo.startswith("molde_") and archivo.endswith(".png"):
            # Extraemos el número del nombre (ej: de "molde_12.png" quitamos "molde_" y ".png" -> nos queda "12")
            numero_str = archivo.replace("molde_", "").replace(".png", "")
            
            try:
                numero_real = int(numero_str)
                ruta = os.path.join(carpeta, archivo)
                molde_img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
                
                if molde_img is not None:
                    # Aseguramos binarización estricta (tinta blanca, fondo negro) para cv2.matchShapes
                    _, molde_bin = cv2.threshold(molde_img, 127, 255, cv2.THRESH_BINARY)
                    moldes[numero_real] = molde_bin
            except ValueError:
                pass # Si hay algún archivo mal nombrado, lo ignoramos
                
    print(f"Diccionario cargado con {len(moldes)} moldes de números.")
    return moldes

def reconocer_numero(molde_extraido, diccionario_moldes):
    mejor_coincidencia = None
    max_similitud = -1

    _, molde_vivo_bin = cv2.threshold(molde_extraido, 127, 255, cv2.THRESH_BINARY)
    
    # EL TRUCO DEL DESLIZAMIENTO:
    # Añadimos 10 píxeles de margen negro. Esto permite que cv2.matchTemplate 
    # "deslice" tu molde por encima de la imagen en vivo y encuentre el encaje perfecto.
    molde_vivo_padded = cv2.copyMakeBorder(molde_vivo_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)

    for numero_real, molde_base in diccionario_moldes.items():
        # TM_CCOEFF_NORMED es el algoritmo rey para emparejar patrones con ruido
        resultado = cv2.matchTemplate(molde_vivo_padded, molde_base, cv2.TM_CCOEFF_NORMED)
        _, similitud, _, _ = cv2.minMaxLoc(resultado) # Devuelve un valor de -1.0 a 1.0
        
        if similitud > max_similitud:
            max_similitud = similitud
            mejor_coincidencia = numero_real

    # Si supera el 55% de similitud estructural estructural, es un match sólido.
    if max_similitud > 0.55:
        return mejor_coincidencia
    return None

# ==============================================================================
# 2. FUNCIONES DE EXTRACCIÓN ROBUSTAS (Solución Validada)
# ==============================================================================
def ordenar_puntos(puntos):
    puntos_ordenados = np.zeros((4, 2), dtype="float32")
    suma = puntos.sum(axis=1)
    puntos_ordenados[0] = puntos[np.argmin(suma)]
    puntos_ordenados[2] = puntos[np.argmax(suma)]
    resta = np.diff(puntos, axis=1)
    puntos_ordenados[1] = puntos[np.argmin(resta)]
    puntos_ordenados[3] = puntos[np.argmax(resta)]
    return puntos_ordenados

def quitar_lineas(binaria):
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    k_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    vert = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_vert)
    hor = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_hor)
    lineas = cv2.bitwise_or(vert, hor)
    lineas = cv2.dilate(lineas, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binaria, cv2.bitwise_not(lineas))

def buscar_digito(binaria_original):
    limpia = quitar_lineas(binaria_original)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(limpia, connectivity=8)
    alto_roi, ancho_roi = limpia.shape
    candidatos = []
    
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (4 < w < 60 and 18 < h < 60 and area > 60): continue
        if x <= 0 or y <= 0 or x + w >= ancho_roi or y + h >= alto_roi: continue
        
        # Filtro de esquinas redondeadas
        centro_x = x + w / 2
        centro_y = y + h / 2
        if centro_x < 35 or centro_y < 35: continue

        mascara = (etiquetas == i).astype(np.uint8) * 255
        anillo = cv2.dilate(mascara, np.ones((7, 7), np.uint8)) - mascara
        fraccion_tinta = np.mean(binaria_original[anillo > 0] > 0)
        if fraccion_tinta > 0.40: continue

        candidatos.append((x, y, w, h))

    if not candidatos: return None, None

    candidatos.sort(key=lambda f: f[0] + f[1])
    x0, y0, w0, h0 = candidatos[0]
    cy0 = y0 + h0 / 2
    grupo = [f for f in candidatos if abs(f[0] - x0) < 50 and abs((f[1] + f[3] / 2) - cy0) < 15]

    x_min = min(f[0] for f in grupo)
    y_min = min(f[1] for f in grupo)
    x_max = max(f[0] + f[2] for f in grupo)
    y_max = max(f[1] + f[3] for f in grupo)

    m = 4
    x1, y1 = max(0, x_min - m), max(0, y_min - m)
    x2, y2 = min(ancho_roi, x_max + m), min(alto_roi, y_max + m)
    recorte = limpia[y1:y2, x1:x2]

    n2, et2, st2, _ = cv2.connectedComponentsWithStats(recorte, connectivity=8)
    solo_digitos = np.zeros_like(recorte)
    for j in range(1, n2):
        if st2[j, cv2.CC_STAT_HEIGHT] >= 15:
            solo_digitos[et2 == j] = 255

    return cv2.resize(solo_digitos, (60, 80)), (x1, y1, x2, y2)

def extraer_numero(carta):
    """
    Recibe la carta de 400x600. Devuelve el molde final (60x80) y la caja, o None.
    """
    for girar in (False, True):
        c = cv2.rotate(carta, cv2.ROTATE_180) if girar else carta
        roi = c[0:170, 0:170]
        g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)

        _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
        _, b3 = cv2.threshold(g, 100, 255, cv2.THRESH_BINARY_INV)

        for b in (b1, b2, b3):
            molde, bbox = buscar_digito(b)
            if molde is not None:
                return molde, bbox
    return None, None

# ==============================================================================
# 4. LÓGICA DEL JUEGO DE MUS (Pares, Juego, Grandes y Chicas)
# ==============================================================================
def evaluar_mus(mano_cruda):
    """
    Evalúa la mano y devuelve una lista de strings para imprimirlas en el HUD.
    """
    if len(mano_cruda) < 4:
        return [] 
    
    mano_cruda = mano_cruda[:4] 

    # 1. Transformar Cerdos (3) en Reyes (12) y Pitos (2) en Ases (1)
    mano = [12 if carta == 3 else 1 if carta == 2 else carta for carta in mano_cruda]

    # Diccionario para nombres visuales
    nombres = {12: "Rey/3", 11: "Caballo", 10: "Sota", 7: "Siete", 6: "Seis", 5: "Cinco", 4: "Cuatro", 1: "As/Pito"}

    # 2. Calcular Pares
    from collections import Counter
    conteos = Counter(mano)
    valores_pares = list(conteos.values())
    
    texto_pares = "Pareja: No"
    if 4 in valores_pares:
        texto_pares = "Pareja: Si | Duples (4 iguales)"
    elif valores_pares.count(2) == 2:
        texto_pares = "Pareja: Si | Duples (2 pares)"
    elif 3 in valores_pares:
        texto_pares = "Pareja: Si | Medias"
    elif 2 in valores_pares:
        texto_pares = "Pareja: Si | Par"

    # 3. Calcular Juego o Punto
    suma_puntos = sum([10 if c in [10, 11, 12] else c for c in mano])
            
    if suma_puntos >= 31:
        texto_juego = f"Juego: Si | Puntos: {suma_puntos}"
        if suma_puntos == 31:
            texto_juego += " (Max!)"
    else:
        texto_juego = f"Juego: No | Punto: {suma_puntos}"

    # 4. Grandes y Chicas (Ordenamos la mano de mayor a menor)
    mano_ordenada = sorted(mano, reverse=True)
    
    # Las dos más grandes (índices 0 y 1)
    g1 = nombres.get(mano_ordenada[0], str(mano_ordenada[0]))
    g2 = nombres.get(mano_ordenada[1], str(mano_ordenada[1]))
    
    # Las dos más chicas (índices -1 y -2, leídas de la más pequeña a la siguiente)
    c1 = nombres.get(mano_ordenada[-1], str(mano_ordenada[-1]))
    c2 = nombres.get(mano_ordenada[-2], str(mano_ordenada[-2]))

    texto_grandes = f"Grandes: {g1}, {g2}"
    texto_chicas  = f"Chicas: {c1}, {c2}"

    # Devolvemos las 4 líneas empaquetadas (El bloque C las dibujará automáticamente en cascada)
    return [texto_pares, texto_juego, texto_grandes, texto_chicas]

# ==============================================================================
# 3. BUCLE PRINCIPAL DE VÍDEO (Con Tracking y Memoria Estabilizadora)
# ==============================================================================
def iniciar_mus_directo():
    diccionario = cargar_diccionario_moldes()
    cap = cv2.VideoCapture(2) 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error al abrir cámara.")
        return

    # --- MEMORIA DEL SISTEMA ---
    # Guardará las cartas que hay en la mesa y sus últimos 15 resultados
    memoria_mesa = [] 
    tiempo_anterior = time.time()
    
    print("\n¡Sistema Activo! Pon las cartas en la mesa.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        suavizado = cv2.GaussianBlur(gris, (15, 15), 0)
        binarizada = cv2.adaptiveThreshold(suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 51, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel)
        
        contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cartas_detectadas_en_mesa = []

        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if 15000 < area < 400000:
                rect = cv2.minAreaRect(contorno)
                (cx, cy), (w_r, h_r), angulo = rect
                
                if w_r == 0 or h_r == 0: continue
                aspect_ratio = float(w_r) / h_r if w_r > h_r else float(h_r) / w_r
                if aspect_ratio > 2.5: continue

                esquinas = cv2.boxPoints(rect)
                esquinas = np.intp(esquinas)
                
                ancho, alto = 400, 600
                esquinas_ordenadas = ordenar_puntos(esquinas)
                
                # Parche anti-deformación (Si está tumbada, la ponemos de pie)
                lado_superior = np.linalg.norm(esquinas_ordenadas[0] - esquinas_ordenadas[1])
                lado_izquierdo = np.linalg.norm(esquinas_ordenadas[0] - esquinas_ordenadas[3])
                if lado_superior > lado_izquierdo:
                    esquinas_ordenadas = np.roll(esquinas_ordenadas, 1, axis=0)
                
                puntos_destino = np.array([[0, 0], [ancho-1, 0], [ancho-1, alto-1], [0, alto-1]], dtype="float32")
                matriz = cv2.getPerspectiveTransform(esquinas_ordenadas, puntos_destino)
                carta_extraida = cv2.warpPerspective(frame, matriz, (ancho, alto))

                # =================================================================
                # NUEVO: TRÍPODE DIGITAL (Filtro de Nitidez / Sharpening)
                # Contrarresta el desenfoque de la cámara en mano y resalta la tinta.
                # =================================================================
                kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                carta_extraida = cv2.filter2D(carta_extraida, -1, kernel_sharpen)

                # 2. Extraer el dígito limpio
                molde_vivo, _ = extraer_numero(carta_extraida)

                numero_identificado = None
                if molde_vivo is not None:
                    numero_identificado = reconocer_numero(molde_vivo, diccionario)
                
                # =================================================================
                # LÓGICA DE MEMORIA (TRACKING ANTI-PARPADEO)
                # =================================================================
                numero_estable = None
                carta_encontrada = False
                
                # Comprobamos si esta carta ya la estábamos viendo en la mesa
                for mem in memoria_mesa:
                    distancia = math.hypot(mem['cx'] - cx, mem['cy'] - cy)
                    if distancia < 100:  # Si el centro coincide (margen de 100px)
                        carta_encontrada = True
                        mem['cx'], mem['cy'] = cx, cy # Actualizamos posición
                        mem['vida'] = 10 # Reseteamos contador para que no se borre
                        
                        if numero_identificado is not None:
                            mem['historial'].append(numero_identificado)
                            # Recordamos solo los últimos 15 frames
                            if len(mem['historial']) > 15:
                                mem['historial'].pop(0)
                                
                        # EL NÚMERO ESTABLE ES LA MODA (El más repetido en el historial)
                        if len(mem['historial']) > 0:
                            numero_estable = Counter(mem['historial']).most_common(1)[0][0]
                        break
                        
                # Si es una carta nueva en la mesa, la registramos
                if not carta_encontrada:
                    hist_inicial = [numero_identificado] if numero_identificado else []
                    memoria_mesa.append({'cx': cx, 'cy': cy, 'historial': hist_inicial, 'vida': 10})
                    numero_estable = numero_identificado

                # =================================================================
                # DIBUJADO EN PANTALLA
                # =================================================================
                if numero_estable is not None:
                    cartas_detectadas_en_mesa.append(numero_estable)
                    cv2.drawContours(frame, [esquinas], -1, (0, 255, 0), 3)
                    
                    posicion_texto = tuple(esquinas_ordenadas[0].astype(int))
                    cv2.putText(frame, str(numero_estable), (posicion_texto[0], posicion_texto[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6)
                    cv2.putText(frame, str(numero_estable), (posicion_texto[0], posicion_texto[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Mantenimiento de memoria (Borrar las cartas que ya hemos quitado de la mesa)
        memoria_viva = []
        for mem in memoria_mesa:
            mem['vida'] -= 1
            if mem['vida'] > 0:
                memoria_viva.append(mem)
        memoria_mesa = memoria_viva

        # C. INTERFAZ DE RENDIMIENTO Y RESULTADO DEL MUS
        tiempo_actual = time.time()
        fps = 1 / (tiempo_actual - tiempo_anterior) if (tiempo_actual - tiempo_anterior) > 0 else 0
        tiempo_anterior = tiempo_actual
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Mesa: {cartas_detectadas_en_mesa}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Lógica de dibujo del "HUD" del Mus
        if len(cartas_detectadas_en_mesa) >= 4:
            lineas_mus = evaluar_mus(cartas_detectadas_en_mesa)
            y_inicial = 130
            
            # Dibujamos cada línea devuelta por la lógica, bajando 40 píxeles por cada una
            for i, linea in enumerate(lineas_mus):
                y_offset = y_inicial + (i * 40)
                # Sombra negra gruesa para que se lea perfecto sobre cualquier fondo
                cv2.putText(frame, linea, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                # Letra verde vibrante encima
                cv2.putText(frame, linea, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            # Mensaje de advertencia si faltan cartas en la mesa
            aviso = f"Esperando 4 cartas... (Detectadas: {len(cartas_detectadas_en_mesa)})"
            cv2.putText(frame, aviso, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(frame, aviso, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Analisis de MUS en Directo', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_mus_directo()