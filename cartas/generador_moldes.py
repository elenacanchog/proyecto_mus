import cv2
import numpy as np
import os

# Carpeta donde se guardan los moldes de cada ejecución (se crea si no existe)
CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'moldes_generados')
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ==============================================================================
# 1. CARGAR Y REDIMENSIONAR LA IMAGEN
# ==============================================================================
ruta_imagen = '../baraja.jpg'
imagen_original = cv2.imread(ruta_imagen)

if imagen_original is None:
    print(f"Error: No se pudo cargar la imagen en {ruta_imagen}. Revisa el nombre y la ruta.")
    exit()

alto_orig, ancho_orig = imagen_original.shape[:2]
nuevo_ancho = 1200
escala = nuevo_ancho / ancho_orig
nuevo_alto = int(alto_orig * escala)
imagen = cv2.resize(imagen_original, (nuevo_ancho, nuevo_alto))
imagen_resultado = imagen.copy()

# ==============================================================================
# 2. PREPROCESADO PARA ENCONTRAR LAS CARTAS
# ==============================================================================
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
binarizada = cv2.adaptiveThreshold(suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 4)
contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


def ordenar_puntos(puntos):
    """Orden: Arriba-Izq, Arriba-Der, Abajo-Der, Abajo-Izq."""
    puntos_ordenados = np.zeros((4, 2), dtype="float32")
    suma = puntos.sum(axis=1)
    puntos_ordenados[0] = puntos[np.argmin(suma)]
    puntos_ordenados[2] = puntos[np.argmax(suma)]
    resta = np.diff(puntos, axis=1)
    puntos_ordenados[1] = puntos[np.argmin(resta)]
    puntos_ordenados[3] = puntos[np.argmax(resta)]
    return puntos_ordenados


# ==============================================================================
# 3. EXTRACCIÓN DEL NÚMERO (nueva versión robusta)
# ==============================================================================
def quitar_lineas(binaria):
    """Borra líneas rectas largas (marco de la carta y borde de la mesa).
    El número es más corto que el kernel, así que sobrevive."""
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    k_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    vert = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_vert)
    hor = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, k_hor)
    lineas = cv2.bitwise_or(vert, hor)
    lineas = cv2.dilate(lineas, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binaria, cv2.bitwise_not(lineas))


def buscar_digito(binaria_original):
    """Recibe una binaria (tinta=blanco). Devuelve (imagen 60x80, bbox) o None."""
    limpia = quitar_lineas(binaria_original)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(limpia, connectivity=8)
    alto_roi, ancho_roi = limpia.shape

    candidatos = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]

        # Tamaño típico de un dígito en la carta de 400x600
        if not (4 < w < 60 and 18 < h < 60 and area > 60):
            continue
        # No puede tocar el borde del recorte
        if x <= 0 or y <= 0 or x + w >= ancho_roi or y + h >= alto_roi:
            continue
        
        # NUEVO: descartar la esquina redondeada de la carta
        centro_x = x + w / 2
        centro_y = y + h / 2
        if centro_x < 35 or centro_y < 35:
            continue

        # Un dígito está rodeado de papel (negro en la binaria).
        # Los restos de mesa/marco están rodeados de blanco -> descartar.
        mascara = (etiquetas == i).astype(np.uint8) * 255
        anillo = cv2.dilate(mascara, np.ones((7, 7), np.uint8)) - mascara
        fraccion_tinta = np.mean(binaria_original[anillo > 0] > 0)
        if fraccion_tinta > 0.40:
            continue

        print(f"candidato: x={x} y={y} w={w} h={h} area={area}")

        candidatos.append((x, y, w, h))

    if not candidatos:
        return None

    # Ancla: la mancha más cercana a la esquina superior izquierda
    candidatos.sort(key=lambda f: f[0] + f[1])
    x0, y0, w0, h0 = candidatos[0]
    cy0 = y0 + h0 / 2

    # Agrupar dígitos de "10", "11", "12": misma altura y cerca en X
    grupo = [f for f in candidatos
             if abs(f[0] - x0) < 50 and abs((f[1] + f[3] / 2) - cy0) < 15]

    x_min = min(f[0] for f in grupo)
    y_min = min(f[1] for f in grupo)
    x_max = max(f[0] + f[2] for f in grupo)
    y_max = max(f[1] + f[3] for f in grupo)

    m = 4
    x1, y1 = max(0, x_min - m), max(0, y_min - m)
    x2, y2 = min(ancho_roi, x_max + m), min(alto_roi, y_max + m)

    # Recortamos de la imagen YA LIMPIA de líneas, para que el molde no lleve restos del marco
    # Recortamos de la imagen YA LIMPIA de líneas
    recorte = limpia[y1:y2, x1:x2]

    # NUEVO: dentro del recorte, nos quedamos solo con las manchas altas (los dígitos).
    # Las líneas finas del marco (2-4 px de alto) se eliminan.
    n2, et2, st2, _ = cv2.connectedComponentsWithStats(recorte, connectivity=8)
    solo_digitos = np.zeros_like(recorte)
    for j in range(1, n2):
        if st2[j, cv2.CC_STAT_HEIGHT] >= 15:
            solo_digitos[et2 == j] = 255

    return cv2.resize(solo_digitos, (60, 80)), (x1, y1, x2, y2)


def extraer_numero(carta):
    """carta: imagen planchada 400x600 BGR. Devuelve (molde 60x80, bbox) o (None, None)."""
    for girar in (False, True):            # 2º intento: esquina inferior derecha (girada 180º)
        c = cv2.rotate(carta, cv2.ROTATE_180) if girar else carta
        roi = c[0:170, 0:170]
        g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3, 3), 0)

        _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 10)
        _, b3 = cv2.threshold(g, 100, 255, cv2.THRESH_BINARY_INV)

        for b in (b1, b2, b3):             # probamos umbrales hasta que uno funcione
            res = buscar_digito(b)
            if res is not None:
                molde, (x1, y1, x2, y2) = res
                if girar:
                    # La caja está en coordenadas de la carta girada 180°.
                    # La devolvemos a coordenadas de la carta original (400x600).
                    return molde, (400 - x2, 600 - y2, 400 - x1, 600 - y1)
                return molde, (x1, y1, x2, y2)
    return None, None


# ==============================================================================
# 4. FILTRAR CARTAS, PLANCHAR Y EXTRAER NÚMEROS
# ==============================================================================
contador_cartas = 0
cartas_planchadas = []

for contorno in contornos:
    area = cv2.contourArea(contorno)

    if 5000 < area < 300000:
        rect = cv2.minAreaRect(contorno)
        esquinas = cv2.boxPoints(rect)
        esquinas = np.intp(esquinas)
        cv2.drawContours(imagen_resultado, [esquinas], -1, (0, 255, 0), 3)

        ancho, alto = 400, 600
        esquinas_ordenadas = ordenar_puntos(esquinas)
        puntos_destino = np.array([[0, 0], [ancho - 1, 0],
                                   [ancho - 1, alto - 1], [0, alto - 1]], dtype="float32")

        matriz_perspectiva = cv2.getPerspectiveTransform(esquinas_ordenadas, puntos_destino)
        carta_extraida = cv2.warpPerspective(imagen, matriz_perspectiva, (ancho, alto))

        carta_vista = cv2.resize(carta_extraida, (200, 300))
        cartas_planchadas.append(carta_vista)

        # --- Extraer el número ---
        molde, bbox = extraer_numero(carta_extraida)

        if molde is not None:
            contador_cartas += 1
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(cartas_planchadas[-1], (x1 // 2, y1 // 2),
                              (x2 // 2, y2 // 2), (0, 255, 0), 1)
            cv2.imshow(f'Numero {contador_cartas}', molde)
            cv2.imwrite(os.path.join(CARPETA_SALIDA, f'carta_generada_{contador_cartas:02d}.png'), molde)
        else:
            print(f"AVISO: no se pudo leer el número de la carta {len(cartas_planchadas)}")
            cv2.rectangle(cartas_planchadas[-1], (0, 0), (199, 299), (0, 0, 255), 3)

print(f"Total de cartas detectadas y planchadas: {len(cartas_planchadas)}")
print(f"Números leídos correctamente: {contador_cartas}")

# ==============================================================================
# 5. VENTANAS DIDÁCTICAS
# ==============================================================================
cv2.imshow('1. Escala de Grises', gris)
cv2.imshow('2. Suavizado', suavizado)
cv2.imshow('3. Binarizada (Buscador de formas)', binarizada)
cv2.imshow('4. Resultado (Contornos Perfectos)', imagen_resultado)

if len(cartas_planchadas) > 0:
    vista_cartas_extraidas = np.hstack(cartas_planchadas)
    cv2.imshow('5. Escaner de Cartas (Perspectiva)', vista_cartas_extraidas)

cv2.waitKey(0)
cv2.destroyAllWindows()