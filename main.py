import cv2
import numpy as np

# 1. Cargar el modelo preentrenado de Viola-Jones para rostros y ojos
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# 1.5. Configurar el ecualizador de luz adaptativo (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 2. Iniciar la captura de vídeo
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

print("Presiona 'q' para salir del programa.")

# Variables para estabilizar las cajas (Filtro de Media Móvil)
historial_caras = []
frames_memoria = 5

# 2.5. Variables para la detección temporal de señas (Máquina de Estados)
contador_31 = 0
contador_ciego = 0
UMBRAL_FRAMES = 7        # Fotogramas seguidos necesarios para confirmar la seña
MAX_CONTADOR = 14        # Tope máximo del contador
UMBRAL_PUPILA = 20       # Bajado a 20% para que tus ojos abiertos (ej. 29% en seg 00:22) nunca den falso cierre

# Variables para la seña de DOS PITOS (Sacar la lengua)
contador_lengua = 0
UMBRAL_LENGUA = 8       # Porcentaje del recuadro de la boca con color de lengua para confirmar el gesto

# Variables para la seña de DUPLES (Levantar las cejas)
contador_duples = 0
base_cejas_y = None          # Memoriza la altura normal de tus cejas en reposo (0 a 100)
UMBRAL_SUBIDA_CEJAS = 7.5    # Cuántos puntos porcentuales deben subir las cejas respecto a tu reposo

# NUEVO: Memoria anatómica de la altura y tamaño normal de tus ojos abiertos (evita saltar a la ceja)
ref_ey = None
ref_eh = None

# 3. Bucle principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el fotograma.")
        break
    
    # Modo espejo para que tu ojo izquierdo esté a la izquierda de la pantalla
    frame = cv2.flip(frame, 1)
    
    # 4. Optimización y corrección de luz
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_clahe = clahe.apply(gray)

    # 5. Detectar rostros
    faces = face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # 6. Analizar cada rostro detectado
    for (x, y, w, h) in faces:

        # --- FILTRO DE SUAVIZADO (MEDIA MÓVIL) ---
        historial_caras.append((x, y, w, h))
        if len(historial_caras) > frames_memoria:
            historial_caras.pop(0)
            
        x = int(sum([c[0] for c in historial_caras]) / len(historial_caras))
        y = int(sum([c[1] for c in historial_caras]) / len(historial_caras))
        w = int(sum([c[2] for c in historial_caras]) / len(historial_caras))
        h = int(sum([c[3] for c in historial_caras]) / len(historial_caras))
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # ---------------------------------------------------------
        # ROI de las CEJAS Y DETECCIÓN DE "DUPLES" (LEVANTAR CEJAS)
        # ---------------------------------------------------------
        x_cejas_inicio = int(x + w * 0.18)
        x_cejas_fin = int(x + w * 0.82)
        y_cejas_inicio = int(y + h * 0.14)
        y_cejas_fin = int(y + h * 0.30)

        # Dibujamos rectángulo morado (255, 0, 180) para la zona de las cejas
        cv2.rectangle(frame, (x_cejas_inicio, y_cejas_inicio), (x_cejas_fin, y_cejas_fin), (255, 0, 180), 2)

        # 1. Recortamos la zona de las cejas de la imagen con luz corregida (gray_clahe)
        roi_gray_cejas = gray_clahe[y_cejas_inicio:y_cejas_fin, x_cejas_inicio:x_cejas_fin]

        if roi_gray_cejas.size > 0:
            # 2. Suavizado Gaussiano para eliminar ruido de la piel antes de binarizar
            cejas_suavizadas = cv2.GaussianBlur(roi_gray_cejas, (5, 5), 0)

            # 3. Binarización dinámica: aislamos los pelos oscuros de la ceja respecto a la piel de la frente
            media_luz_cejas = np.mean(cejas_suavizadas)
            _, cejas_binarias = cv2.threshold(cejas_suavizadas, int(media_luz_cejas * 0.78), 255, cv2.THRESH_BINARY_INV)

            # --- NUEVO BLINDAJE: USAR CONTORNOS PARA BORRAR PELOS Y ARRUGAS ---
            # a) Pequeña apertura para desconectar posibles pelos que toquen la ceja
            kernel_cejas = np.ones((3, 3), np.uint8)
            cejas_limpias = cv2.morphologyEx(cejas_binarias, cv2.MORPH_OPEN, kernel_cejas)

            # b) Buscamos todas las formas blancas de la caja morada
            contornos_cejas, _ = cv2.findContours(cejas_limpias, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Creamos un lienzo negro vacío
            cejas_finales = np.zeros_like(cejas_binarias)

            if contornos_cejas:
                # c) Ordenamos los contornos por tamaño (de mayor a menor) y nos quedamos SOLO con los 2 primeros
                contornos_ordenados = sorted(contornos_cejas, key=cv2.contourArea, reverse=True)[:2]
                
                # Dibujamos nuestras 2 cejas ganadoras en el lienzo negro
                cv2.drawContours(cejas_finales, contornos_ordenados, -1, 255, cv2.FILLED)

            # 4. Calculamos el Centro de Gravedad Vertical usando SOLO las 2 cejas limpias
            filas_blancas, cols_blancas = np.where(cejas_finales > 0)

            if len(filas_blancas) > 30: # Aseguramos que haya suficientes píxeles de ceja detectados
                # Altura media normalizada de 0 (techo de la caja) a 100 (suelo de la caja)
                altura_actual_cejas = (np.mean(filas_blancas) / cejas_binarias.shape[0]) * 100.0

                # Inicializamos la altura de reposo en el primer fotograma
                if base_cejas_y is None:
                    base_cejas_y = altura_actual_cejas

                # Calculamos cuánto han subido las cejas (como Y=0 está arriba, al subir las cejas el valor disminuye)
                subida_cejas = base_cejas_y - altura_actual_cejas

                # Dibujamos en pantalla una línea horizontal verde donde el ordenador detecta el centro de tus cejas
                y_linea_ceja = y_cejas_inicio + int((altura_actual_cejas / 100.0) * (y_cejas_fin - y_cejas_inicio))
                cv2.line(frame, (x_cejas_inicio, y_linea_ceja), (x_cejas_fin, y_linea_ceja), (0, 255, 0), 2)

                # Mostramos el valor de subida encima de la caja morada
                cv2.putText(frame, f"Cejas Subida: {subida_cejas:.1f}", (x_cejas_inicio, y_cejas_inicio - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 180), 1)

                # 5. Máquina de estados para Duples y actualización de la línea base
                if subida_cejas >= UMBRAL_SUBIDA_CEJAS:
                    # Las cejas han subido claramente -> Sumamos al contador de Duples (y NO actualizamos la base)
                    contador_duples = min(MAX_CONTADOR, contador_duples + 2)
                else:
                    # Estás en reposo -> Bajamos el contador y actualizamos suavemente tu altura base (95% memoria + 5% actual)
                    contador_duples = max(0, contador_duples - 2)
                    base_cejas_y = (base_cejas_y * 0.95) + (altura_actual_cejas * 0.05)

            # Ampliamos la máscara de las cejas para verla en una 6ª ventana de depuración
            cejas_binarias_ampliadas = cv2.resize(cejas_finales, (240, 90))

            # 6. Mostrar el texto de la seña si supera los fotogramas seguidos
            if contador_duples >= UMBRAL_FRAMES:
                cv2.putText(frame, "Sena duples (cejas) detectada", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 180), 3)

        # ---------------------------------------------------------
        # EXTRACCIÓN GEOMÉTRICA DE REGIONES DE INTERÉS (ROI)
        # ---------------------------------------------------------

        # ROI de los OJOS: Bajamos el techo a 0.29 para dejar las cejas físicamente fuera de la caja azul
        x_ojos_inicio = int(x + w * 0.15)
        x_ojos_fin = int(x + w * 0.85)
        y_ojos_inicio = int(y + h * 0.29)
        y_ojos_fin = int(y + h * 0.53)
        
        cv2.rectangle(frame, (x_ojos_inicio, y_ojos_inicio), (x_ojos_fin, y_ojos_fin), (255, 0, 0), 2)
        cv2.putText(frame, 'Ojos', (x_ojos_inicio, y_ojos_inicio - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # --- DETECCIÓN DE OJOS DENTRO DE LA CAJA AZUL ---
        roi_gray_ojos = gray_clahe[y_ojos_inicio:y_ojos_fin, x_ojos_inicio:x_ojos_fin]
        ancho_roi_ojos = x_ojos_fin - x_ojos_inicio
        
        tam_max_ojo = int(w * 0.22)
        ojos_detectados = eye_cascade.detectMultiScale(
            roi_gray_ojos, 
            scaleFactor=1.1, 
            minNeighbors=6, 
            minSize=(22, 22),
            maxSize=(tam_max_ojo, tam_max_ojo)
        )

        # --- NUEVO: FILTRO ANTI-CEJA (VALIDACIÓN ANATÓMICA) ---
        # Separamos el mejor candidato de ojo izquierdo y derecho
        cand_izq = None
        cand_der = None
        for (ex, ey, ew, eh) in ojos_detectados:
            if (ex + ew // 2) < (ancho_roi_ojos // 2):
                # Si hay varios en la izquierda, nos quedamos con el que esté más abajo (más lejos de la ceja)
                if cand_izq is None or (ey + eh // 2) > (cand_izq[1] + cand_izq[3] // 2):
                    cand_izq = (ex, ey, ew, eh)
            else:
                if cand_der is None or (ey + eh // 2) > (cand_der[1] + cand_der[3] // 2):
                    cand_der = (ex, ey, ew, eh)

        # Si vemos los DOS ojos a la vez y están alineados en horizontal, actualizamos la memoria de tus ojos abiertos
        if cand_izq is not None and cand_der is not None:
            if abs(cand_izq[1] - cand_der[1]) < int(h * 0.05):
                ey_medio = (cand_izq[1] + cand_der[1]) / 2.0
                eh_medio = (cand_izq[3] + cand_der[3]) / 2.0
                ref_ey = ey_medio if ref_ey is None else (ref_ey * 0.8 + ey_medio * 0.2)
                ref_eh = eh_medio if ref_eh is None else (ref_eh * 0.8 + eh_medio * 0.2)

        # Comprobamos que ningún candidato haya saltado hacia arriba (ceja) ni haya crecido de golpe
        ojos = []
        for cand in [cand_izq, cand_der]:
            if cand is not None:
                ex, ey, ew, eh = cand
                if ref_ey is not None and ref_eh is not None:
                    # Si la caja sube demasiado respecto a tu ojo normal o es un >20% más grande, es la ceja: la ignoramos
                    if ey < (ref_ey - ref_eh * 0.25) or eh > (ref_eh * 1.20):
                        continue
                ojos.append(cand)

        # Lienzos en negro para mostrar AMBOS ojos (Izquierdo | Derecho) en la Ventana 4
        vista_ojo_izq = np.zeros((150, 150), dtype=np.uint8)
        vista_ojo_der = np.zeros((150, 150), dtype=np.uint8)

        porcentaje_izq = 0
        porcentaje_der = 0

        # Bucle para analizar cada ojo válido
        for (ex, ey, ew, eh) in ojos:
            centro_x = x_ojos_inicio + ex
            centro_y = y_ojos_inicio + ey

            # Recorte central del ojo
            y1_int = ey + int(eh * 0.25)
            y2_int = ey + int(eh * 0.65)
            x1_int = ex + int(ew * 0.25)
            x2_int = ex + int(ew * 0.75)

            ojo_centro_gris = roi_gray_ojos[y1_int:y2_int, x1_int:x2_int]
            ojo_completo_gris = roi_gray_ojos[ey:ey+eh, ex:ex+ew]

            if ojo_centro_gris.size == 0:
                continue

            media_luz_ojo = np.mean(ojo_completo_gris)
            umbral_dinamico = int(media_luz_ojo * 0.75)

            _, ojo_binario = cv2.threshold(ojo_centro_gris, umbral_dinamico, 255, cv2.THRESH_BINARY_INV)

            pixeles_oscuros = cv2.countNonZero(ojo_binario)
            total_pixeles = ojo_centro_gris.shape[0] * ojo_centro_gris.shape[1]
            porcentaje_oscuro = int((pixeles_oscuros / total_pixeles) * 100)

            if (ex + ew // 2) < (ancho_roi_ojos // 2):
                porcentaje_izq = max(porcentaje_izq, porcentaje_oscuro)
                vista_ojo_izq = cv2.resize(ojo_binario, (150, 150))
            else:
                porcentaje_der = max(porcentaje_der, porcentaje_oscuro)
                vista_ojo_der = cv2.resize(ojo_binario, (150, 150))

            if porcentaje_oscuro >= UMBRAL_PUPILA:
                color_caja_ojo = (0, 255, 255) # Amarillo = Abierto
            else:
                color_caja_ojo = (0, 140, 255) # Naranja = Cerrado

            cv2.rectangle(frame, (centro_x, centro_y), (centro_x + ew, centro_y + eh), color_caja_ojo, 2)
            cv2.rectangle(frame, (x_ojos_inicio + x1_int, y_ojos_inicio + y1_int), 
                          (x_ojos_inicio + x2_int, y_ojos_inicio + y2_int), (255, 255, 0), 1)
            cv2.putText(frame, f"Pxs:{pixeles_oscuros} ({porcentaje_oscuro}%)", (centro_x, centro_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja_ojo, 1)

        # Unimos la vista del ojo izquierdo y derecho separados por una línea blanca para la Ventana 4
        linea_separadora = np.ones((150, 4), dtype=np.uint8) * 255
        ojo_binario_ampliado = np.hstack((vista_ojo_izq, linea_separadora, vista_ojo_der))

        # ---------------------------------------------------------
        # 6.5. LÓGICA DE SEÑAS (31 Y CIEGO)
        # ---------------------------------------------------------
        izq_abierto = porcentaje_izq >= UMBRAL_PUPILA
        der_abierto = porcentaje_der >= UMBRAL_PUPILA

        if izq_abierto and der_abierto:
            # Ambos ojos abiertos -> Reseteamos rápido los contadores
            contador_31 = max(0, contador_31 - 3)
            contador_ciego = max(0, contador_ciego - 3)
        elif not izq_abierto and not der_abierto:
            # Ninguno de los dos ojos está abierto -> Seña Ciego
            contador_ciego = min(MAX_CONTADOR, contador_ciego + 2)
            contador_31 = max(0, contador_31 - 3)
        else:
            # Uno abierto y el otro cerrado -> Seña 31
            contador_31 = min(MAX_CONTADOR, contador_31 + 2)
            contador_ciego = max(0, contador_ciego - 2)

        # Imprimir el mensaje ARRIBA A LA IZQUIERDA
        if contador_ciego >= UMBRAL_FRAMES:
            cv2.putText(frame, "Sena ciego detectada", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        elif contador_31 >= UMBRAL_FRAMES:
            cv2.putText(frame, "Sena 31 detectada", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # ---------------------------------------------------------
        # ROI de la BOCA Y DETECCIÓN DE "DOS PITOS" (SACAR LA LENGUA)
        # ---------------------------------------------------------
        x_boca_inicio = int(x + w * 0.25)
        x_boca_fin = int(x + w * 0.75)
        y_boca_inicio = int(y + h * 0.65)
        y_boca_fin = int(y + h * 0.95) # Cortamos en 0.95 para no meter el borde de la barbilla/cuello
        
        cv2.rectangle(frame, (x_boca_inicio, y_boca_inicio), (x_boca_fin, y_boca_fin), (0, 0, 255), 2)

        # 1. Recortamos la zona de la boca de la imagen ORIGINAL A COLOR (frame)
        roi_boca_color = frame[y_boca_inicio:y_boca_fin, x_boca_inicio:x_boca_fin]

        if roi_boca_color.size > 0:
            # 2. Convertimos el recorte de BGR al espacio de color HSV
            hsv_boca = cv2.cvtColor(roi_boca_color, cv2.COLOR_BGR2HSV)

            # 3. Definimos los rangos de color HSV (buscamos tonos rosados/rojizos)
            bajo_rojo1 = np.array([0, 110, 70])
            alto_rojo1 = np.array([12, 255, 255])
            bajo_rojo2 = np.array([165, 110, 70])
            alto_rojo2 = np.array([179, 255, 255])

            mascara1 = cv2.inRange(hsv_boca, bajo_rojo1, alto_rojo1)
            mascara2 = cv2.inRange(hsv_boca, bajo_rojo2, alto_rojo2)
            mascara_lengua = cv2.bitwise_or(mascara1, mascara2)

            # 4. EL TRUCO CLÁSICO: APERTURA (Borra líneas finas como los labios)
            # MORPH_OPEN primero erosiona (encoge y borra lo pequeño) y luego dilata lo que sobrevive.
            # Un kernel de 5x5 se comerá el ruido de la barbilla y la línea fina de los labios cerrados.
            kernel_apertura = np.ones((5, 5), np.uint8)
            mascara_lengua = cv2.morphologyEx(mascara_lengua, cv2.MORPH_OPEN, kernel_apertura)

            # 5. ENCONTRAR CONTORNOS (BLOBS / FORMAS)
            # Buscamos agrupaciones continuas de píxeles en lugar de contar píxeles sueltos.
            contornos, _ = cv2.findContours(mascara_lengua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            porcentaje_lengua = 0
            if contornos:
                # Nos quedamos EXCLUSIVAMENTE con la mancha más grande que haya en la caja
                contorno_mas_grande = max(contornos, key=cv2.contourArea)
                area_lengua = cv2.contourArea(contorno_mas_grande)
                
                # Calculamos qué porcentaje de la caja roja ocupa ESA ÚNICA MANCHA
                total_pixeles_boca = mascara_lengua.shape[0] * mascara_lengua.shape[1]
                porcentaje_lengua = int((area_lengua / total_pixeles_boca) * 100)

                # Limpiamos la máscara entera a negro y dibujamos SOLO nuestra mancha ganadora de color blanco
                # Esto evita que el ruido residual afecte a la imagen que ves en la Ventana 5
                mascara_lengua = np.zeros_like(mascara_lengua)
                cv2.drawContours(mascara_lengua, [contorno_mas_grande], -1, 255, cv2.FILLED)
            
            # Mostramos el porcentaje junto al texto de la boca
            cv2.putText(frame, f"Boca ({porcentaje_lengua}%)", (x_boca_inicio, y_boca_inicio - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Ampliamos la máscara limpia para verla en la 5ª ventana
            boca_binaria_ampliada = cv2.resize(mascara_lengua, (200, 120))

            # 7. Máquina de estados para la seña de Dos Pitos (Lengua)
            if porcentaje_lengua >= UMBRAL_LENGUA:
                contador_lengua = min(MAX_CONTADOR, contador_lengua + 2)
            else:
                contador_lengua = max(0, contador_lengua - 2)

            # Si supera los fotogramas seguidos, imprimimos la seña en pantalla (justo debajo de donde salen las de ojos)
            if contador_lengua >= UMBRAL_FRAMES:
                cv2.putText(frame, "Sena dos pitos (lengua) detectada", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 3)

    # 7. Mostrar el fotograma procesado en múltiples ventanas
    cv2.imshow('1. Escala de Grises (Sin corregir)', gray)
    cv2.imshow('2. Filtro CLAHE (Luz corregida)', gray_clahe)
    cv2.imshow('3. Deteccion de Rostro - Presiona Q para salir', frame)
    
    # 7.5 Mostrar la ventana del filtro con AMBOS ojos (Izquierdo | Derecho)
    if 'ojo_binario_ampliado' in locals() and ojo_binario_ampliado is not None:
        cv2.imshow('4. Filtro Blanco y Negro (Ojo Izq | Ojo Der)', ojo_binario_ampliado)

    # 7.6 Mostrar la ventana del filtro HSV de la boca (Lengua)
    if 'boca_binaria_ampliado' in locals() or ('boca_binaria_ampliada' in locals() and boca_binaria_ampliada is not None):
        cv2.imshow('5. Filtro HSV (Boca - Lengua)', boca_binaria_ampliada)

    # 7.7 Mostrar la ventana del filtro binario de las cejas (Duples)
    if 'cejas_binarias_ampliadas' in locals() and cejas_binarias_ampliadas is not None:
        cv2.imshow('6. Filtro Binario (Cejas - Duples)', cejas_binarias_ampliadas)

    # 8. Condición de salida
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Limpieza
cap.release()
cv2.destroyAllWindows()