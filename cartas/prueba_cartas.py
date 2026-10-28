import cv2
import numpy as np

# 1. Cargar la imagen de las espadas para crear los moldes
ruta_imagen = '../baraja.jpg'
imagen_original = cv2.imread(ruta_imagen)

if imagen_original is None:
    print(f"Error: No se pudo cargar la imagen en {ruta_imagen}. Revisa el nombre y la ruta.")
    exit()

# --- NUEVO: REDIMENSIONAR LA FOTO GIGANTE DEL MÓVIL ---
# Si la imagen es muy grande, la reducimos a un ancho manejable (ej. 1200px) 
# manteniendo la proporción para que quepa en tu pantalla.
alto_orig, ancho_orig = imagen_original.shape[:2]
nuevo_ancho = 1200
escala = nuevo_ancho / ancho_orig
nuevo_alto = int(alto_orig * escala)
imagen = cv2.resize(imagen_original, (nuevo_ancho, nuevo_alto))

# Hacemos una copia para dibujar el resultado final a color sin estropear la original
imagen_resultado = imagen.copy()

# 2. Paso 1: Convertir a Escala de Grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# 3. Paso 2: Suavizado Gaussiano (Borra texturas diminutas)
suavizado = cv2.GaussianBlur(gris, (5, 5), 0)

# 4. Paso 3: Binarización Adaptativa (Robusta ante sombras e iluminación irregular)
# cv2.adaptiveThreshold calcula el umbral para pequeñas regiones, ignorando sombras globales.
# Parámetros: 255 (valor max), ADAPTIVE_THRESH_GAUSSIAN_C (método), THRESH_BINARY_INV (tipo), 
# 21 (tamaño del bloque), 4 (constante a restar de la media para afinar el borde).
binarizada = cv2.adaptiveThreshold(suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 4)

# 5. Paso 4: Búsqueda de Contornos
contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- NUEVA FUNCIÓN MATEMÁTICA PARA EL PASO 2 ---
def ordenar_puntos(puntos):
    """
    Ordena las 4 esquinas en este orden estricto: 
    Arriba-Izquierda, Arriba-Derecha, Abajo-Derecha, Abajo-Izquierda.
    Es obligatorio para que al "planchar" la carta no quede del revés o espejada.
    """
    puntos_ordenados = np.zeros((4, 2), dtype="float32")
    # Sumar (x+y): el menor es Arriba-Izquierda, el mayor es Abajo-Derecha
    suma = puntos.sum(axis=1)
    puntos_ordenados[0] = puntos[np.argmin(suma)]
    puntos_ordenados[2] = puntos[np.argmax(suma)]
    # Restar (y-x): el menor es Arriba-Derecha, el mayor es Abajo-Izquierda
    resta = np.diff(puntos, axis=1)
    puntos_ordenados[1] = puntos[np.argmin(resta)]
    puntos_ordenados[3] = puntos[np.argmax(resta)]
    return puntos_ordenados

# 6. Filtrar, Dibujar y "Planchar" (Warp Perspective)
contador_cartas = 0
cartas_planchadas = [] 

for contorno in contornos:
    area = cv2.contourArea(contorno)
    
    # Filtro seguro de tamaño para atrapar solo las cartas (ignorando ruido y mesa)
    if 5000 < area < 300000: 
        rect = cv2.minAreaRect(contorno)
        esquinas = cv2.boxPoints(rect)
        esquinas = np.intp(esquinas)
        
        # Dibujar el contorno verde en la imagen original
        cv2.drawContours(imagen_resultado, [esquinas], -1, (0, 255, 0), 3)

        # 2. EL ESCÁNER PERFECTO (Sin rotaciones inventadas)
        # Forzamos una alta resolución (400x600) para tener precisión en los números
        ancho, alto = 400, 600
        esquinas_ordenadas = ordenar_puntos(esquinas)
        
        puntos_destino = np.array([
            [0, 0],               
            [ancho - 1, 0],       
            [ancho - 1, alto - 1],
            [0, alto - 1]         
        ], dtype="float32")

        # Planchado matemático
        matriz_perspectiva = cv2.getPerspectiveTransform(esquinas_ordenadas, puntos_destino)
        carta_extraida = cv2.warpPerspective(imagen, matriz_perspectiva, (ancho, alto))
        
        # Guardamos una versión pequeña solo para la visualización en la Ventana 5
        carta_vista = cv2.resize(carta_extraida, (200, 300))
        cartas_planchadas.append(carta_vista)

        # ==============================================================================
        # 3. BUSCADOR DINÁMICO BASADO EN MORFOLOGÍA (Ignora la línea negra automáticamente)
        # ==============================================================================
        
        # Cogemos una "Zona Segura" enorme en la esquina (160x160) desde el borde absoluto (0,0).
        # Esto atrapará la raya negra, el número y el palo, pero luego lo filtraremos.
        zona_segura = carta_extraida[0:160, 0:160]
        zona_gris = cv2.cvtColor(zona_segura, cv2.COLOR_BGR2GRAY)
        
        # Binarizamos para que la tinta (número/raya) sea blanca y la carta negra
        _, zona_bin = cv2.threshold(zona_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        
        
        # Buscamos todas las manchas de tinta en esa esquina
        contornos_zona, _ = cv2.findContours(zona_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        formas_limpias = []
        for c in contornos_zona:
            x, y, w, h = cv2.boundingRect(c)
            
            # EL FILTRO MAESTRO (Ajustado quirúrgicamente):
            # 1. w > 90 o h > 100: Es gigante (ES LA RAYA NEGRA "L"). A la basura.
            # 2. w < 5 o h < 15: Es microscópico (polvo). A la basura.
            # 3. NUEVO: x > 90: Si la forma empieza más allá del píxel 90, está 
            #    en el centro de la carta (como la copa del 2). ¡A la basura!
            if 5 < w < 90 and 15 < h < 100 and x < 90:
                formas_limpias.append({'c': c, 'x': x, 'y': y, 'w': w, 'h': h})
        
        if formas_limpias:
            contador_cartas += 1
            
            # ORDENACIÓN MAGISTRAL (Eje X): Ordenamos de izquierda a derecha.
            # Como hemos filtrado la raya negra, la forma más a la izquierda ES EL NÚMERO.
            formas_limpias.sort(key=lambda item: item['x'])
            
            forma_principal = formas_limpias[0]
            x_pared_izda = forma_principal['x']
            y_centro_num = forma_principal['y'] + (forma_principal['h'] / 2)
            
            partes_numero = []
            for f in formas_limpias:
                # Agrupamos formas que cumplan dos condiciones lógicas:
                # 1. Que no estén muy lejos hacia la derecha (margen de 45px desde el primer dígito).
                #    Esto atrapa el "2" del "12" pero ignora dibujos en el centro de la carta.
                # 2. Que estén más o menos a la misma altura (alineados horizontalmente).
                #    Esto evita coger miniaturas de palos que estén justo debajo del número.
                centro_f = f['y'] + (f['h'] / 2)
                if (f['x'] - x_pared_izda) < 45 and abs(centro_f - y_centro_num) < 30:
                    partes_numero.append(f)
            
            # Calculamos la caja final que envuelve al dígito (o dígitos) perfectamente
            x_min = min([f['x'] for f in partes_numero])
            y_min = min([f['y'] for f in partes_numero])
            x_max = max([f['x'] + f['w'] for f in partes_numero])
            y_max = max([f['y'] + f['h'] for f in partes_numero])
            
            # Añadimos un pequeño margen de 4 píxeles
            margen = 4
            x_min, y_min = max(0, x_min - margen), max(0, y_min - margen)
            x_max, y_max = min(160, x_max + margen), min(160, y_max + margen)
            
            cv2.rectangle(cartas_planchadas[-1], (int(x_min/2), int(y_min/2)), 
                          (int(x_max/2), int(y_max/2)), (0, 255, 0), 1)

            recorte_numero = zona_bin[y_min:y_max, x_min:x_max]
            num_estandarizado = cv2.resize(recorte_numero, (60, 80))
            
            cv2.imshow(f'Numero {contador_cartas}', num_estandarizado)
            cv2.imwrite(f'molde_carta_{contador_cartas}.png', num_estandarizado)

print(f"Total de cartas detectadas y planchadas: {contador_cartas}")

# 7. Mostrar las ventanas didácticas paso a paso
cv2.imshow('1. Escala de Grises', gris)
cv2.imshow('2. Suavizado', suavizado)
cv2.imshow('3. Binarizada (Buscador de formas)', binarizada)
cv2.imshow('4. Resultado (Contornos Perfectos)', imagen_resultado)

# 8. NUEVA VENTANA: Mostrar todas las cartas extraídas y planchadas
if len(cartas_planchadas) > 0:
    # Juntamos todas las cartas planchadas una al lado de la otra (Horizontal Stack)
    vista_cartas_extraidas = np.hstack(cartas_planchadas)
    cv2.imshow('5. Escaner de Cartas (Perspectiva)', vista_cartas_extraidas)

cv2.waitKey(0)
cv2.destroyAllWindows()