# ==============================================================================
# detector_senas.py
# Motor de lógica y visión artificial para la detección de gestos del Mus.
# ==============================================================================
import cv2
import numpy as np
import config_senas as cfg

class DetectorSenasMus:
    def __init__(self):
        # 1. Cargar el modelo preentrenado de Viola-Jones para rostros y ojos
        self.face_cascade = cv2.CascadeClassifier(cfg.RUTA_CARA)
        self.eye_cascade = cv2.CascadeClassifier(cfg.RUTA_OJOS)
        
        # 1.5. Configurar el ecualizador de luz adaptativo (CLAHE)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Variables para estabilizar las cajas (Filtro de Media Móvil)
        self.historial_caras = []
        
        # Variables de la máquina de estados y memoria anatómica
        self.contador_31 = 0
        self.contador_ciego = 0
        self.contador_lengua = 0
        self.contador_duples = 0
        
        self.base_cejas_y = None # Memoriza la altura normal de tus cejas en reposo (0 a 100)
        self.ref_ey = None       # Memoria anatómica de la altura normal de tus ojos 
        self.ref_eh = None       # Memoria anatómica del tamaño normal de tus ojos
        
        self.textos_senas_activas = [] # Lista para guardar qué textos imprimir en pantalla

    def procesar_fotograma(self, frame):
        """Método principal. Recibe un frame, lo analiza y devuelve ventanas y resultados."""
        self.textos_senas_activas = [] # Reseteamos los textos en cada fotograma nuevo
        
        # 4. Optimización y corrección de luz
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_clahe = self.clahe.apply(gray)

        # 5. Detectar rostros
        faces = self.face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Diccionario para almacenar las ventanas de depuración y pasarlas al main_senas.py
        ventanas = {'gray': gray, 'gray_clahe': gray_clahe, 'cejas_bin': None, 'ojos_bin': None, 'boca_bin': None}

        # 6. Analizar cada rostro detectado
        for (x, y, w, h) in faces:

            # --- FILTRO DE SUAVIZADO (MEDIA MÓVIL) ---
            self.historial_caras.append((x, y, w, h))
            if len(self.historial_caras) > cfg.FRAMES_MEMORIA:
                self.historial_caras.pop(0)
                
            x = int(sum([c[0] for c in self.historial_caras]) / len(self.historial_caras))
            y = int(sum([c[1] for c in self.historial_caras]) / len(self.historial_caras))
            w = int(sum([c[2] for c in self.historial_caras]) / len(self.historial_caras))
            h = int(sum([c[3] for c in self.historial_caras]) / len(self.historial_caras))
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Ejecutar análisis modulares
            ventanas['cejas_bin'] = self.analizar_cejas(frame, gray_clahe, x, y, w, h)
            ventanas['ojos_bin'] = self.analizar_ojos(frame, gray_clahe, x, y, w, h)
            ventanas['boca_bin'] = self.analizar_lengua(frame, x, y, w, h)

        return frame, ventanas, self.textos_senas_activas

    def analizar_cejas(self, frame, gray_clahe, x, y, w, h):
        """ ROI de las CEJAS Y DETECCIÓN DE 'DUPLES' (LEVANTAR CEJAS) """
        x_cejas_inicio = int(x + w * 0.18)
        x_cejas_fin = int(x + w * 0.82)
        y_cejas_inicio = int(y + h * 0.14)
        y_cejas_fin = int(y + h * 0.30)

        # Dibujamos rectángulo morado (255, 0, 180) para la zona de las cejas
        cv2.rectangle(frame, (x_cejas_inicio, y_cejas_inicio), (x_cejas_fin, y_cejas_fin), (255, 0, 180), 2)

        # 1. Recortamos la zona de las cejas de la imagen con luz corregida
        roi_gray_cejas = gray_clahe[y_cejas_inicio:y_cejas_fin, x_cejas_inicio:x_cejas_fin]
        cejas_binarias_ampliadas = None

        if roi_gray_cejas.size > 0:
            # 2. Suavizado Gaussiano para eliminar ruido de la piel antes de binarizar
            cejas_suavizadas = cv2.GaussianBlur(roi_gray_cejas, (5, 5), 0)

            # 3. Binarización dinámica: aislamos los pelos oscuros respecto a la piel de la frente
            media_luz_cejas = np.mean(cejas_suavizadas)
            _, cejas_binarias = cv2.threshold(cejas_suavizadas, int(media_luz_cejas * 0.78), 255, cv2.THRESH_BINARY_INV)

            # --- BLINDAJE: USAR CONTORNOS PARA BORRAR PELOS Y ARRUGAS ---
            # a) Pequeña apertura para desconectar posibles pelos que toquen la ceja
            kernel_cejas = np.ones((3, 3), np.uint8)
            cejas_limpias = cv2.morphologyEx(cejas_binarias, cv2.MORPH_OPEN, kernel_cejas)

            # b) Buscamos todas las formas blancas de la caja morada
            contornos_cejas, _ = cv2.findContours(cejas_limpias, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Creamos un lienzo negro vacío
            cejas_finales = np.zeros_like(cejas_binarias)

            if contornos_cejas:
                # c) Ordenamos los contornos por tamaño y nos quedamos SOLO con los 2 primeros
                contornos_ordenados = sorted(contornos_cejas, key=cv2.contourArea, reverse=True)[:2]
                cv2.drawContours(cejas_finales, contornos_ordenados, -1, 255, cv2.FILLED)

            # 4. Calculamos el Centro de Gravedad Vertical usando SOLO las 2 cejas limpias
            filas_blancas, cols_blancas = np.where(cejas_finales > 0)

            if len(filas_blancas) > 30: # Aseguramos que haya suficientes píxeles detectados
                # Altura media normalizada de 0 (techo) a 100 (suelo)
                altura_actual_cejas = (np.mean(filas_blancas) / cejas_binarias.shape[0]) * 100.0

                # Inicializamos la altura de reposo en el primer fotograma
                if self.base_cejas_y is None:
                    self.base_cejas_y = altura_actual_cejas

                # Calculamos cuánto han subido (como Y=0 está arriba, al subir el valor disminuye)
                subida_cejas = self.base_cejas_y - altura_actual_cejas

                # Línea horizontal verde donde el ordenador detecta el centro
                y_linea_ceja = y_cejas_inicio + int((altura_actual_cejas / 100.0) * (y_cejas_fin - y_cejas_inicio))
                cv2.line(frame, (x_cejas_inicio, y_linea_ceja), (x_cejas_fin, y_linea_ceja), (0, 255, 0), 2)
                cv2.putText(frame, f"Cejas Subida: {subida_cejas:.1f}", (x_cejas_inicio, y_cejas_inicio - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 180), 1)

                # 5. Máquina de estados para Duples y actualización de línea base
                if subida_cejas >= cfg.UMBRAL_SUBIDA_CEJAS:
                    self.contador_duples = min(cfg.MAX_CONTADOR, self.contador_duples + 2)
                else:
                    self.contador_duples = max(0, self.contador_duples - 2)
                    self.base_cejas_y = (self.base_cejas_y * 0.95) + (altura_actual_cejas * 0.05)

            # 6. Mostrar el texto si supera los fotogramas
            if self.contador_duples >= cfg.UMBRAL_FRAMES:
                self.textos_senas_activas.append("Sena duples (cejas) detectada")
                
            # Ampliamos la máscara para depuración
            cejas_binarias_ampliadas = cv2.resize(cejas_finales, (240, 90))

        return cejas_binarias_ampliadas

    def analizar_ojos(self, frame, gray_clahe, x, y, w, h):
        """ EXTRACCIÓN GEOMÉTRICA DE OJOS Y DETECCIÓN DE 31 / CIEGO """
        # Bajamos el techo a 0.29 para dejar las cejas físicamente fuera de la caja azul
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
        ojos_detectados = self.eye_cascade.detectMultiScale(
            roi_gray_ojos, scaleFactor=1.1, minNeighbors=6, minSize=(22, 22), maxSize=(tam_max_ojo, tam_max_ojo)
        )

        # --- FILTRO ANTI-CEJA (VALIDACIÓN ANATÓMICA) ---
        cand_izq = None
        cand_der = None
        for (ex, ey, ew, eh) in ojos_detectados:
            if (ex + ew // 2) < (ancho_roi_ojos // 2):
                # Nos quedamos con el que esté más abajo (más lejos de la ceja)
                if cand_izq is None or (ey + eh // 2) > (cand_izq[1] + cand_izq[3] // 2):
                    cand_izq = (ex, ey, ew, eh)
            else:
                if cand_der is None or (ey + eh // 2) > (cand_der[1] + cand_der[3] // 2):
                    cand_der = (ex, ey, ew, eh)

        # Si vemos los DOS ojos a la vez, actualizamos la memoria anatómica
        if cand_izq is not None and cand_der is not None:
            if abs(cand_izq[1] - cand_der[1]) < int(h * 0.05):
                ey_medio = (cand_izq[1] + cand_der[1]) / 2.0
                eh_medio = (cand_izq[3] + cand_der[3]) / 2.0
                self.ref_ey = ey_medio if self.ref_ey is None else (self.ref_ey * 0.8 + ey_medio * 0.2)
                self.ref_eh = eh_medio if self.ref_eh is None else (self.ref_eh * 0.8 + eh_medio * 0.2)

        # Comprobamos que ningún candidato haya saltado hacia arriba (ceja)
        ojos = []
        for cand in [cand_izq, cand_der]:
            if cand is not None:
                ex, ey, ew, eh = cand
                if self.ref_ey is not None and self.ref_eh is not None:
                    if ey < (self.ref_ey - self.ref_eh * 0.25) or eh > (self.ref_eh * 1.20):
                        continue
                ojos.append(cand)

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

            if porcentaje_oscuro >= cfg.UMBRAL_PUPILA:
                color_caja_ojo = (0, 255, 255) # Amarillo = Abierto
            else:
                color_caja_ojo = (0, 140, 255) # Naranja = Cerrado

            cv2.rectangle(frame, (centro_x, centro_y), (centro_x + ew, centro_y + eh), color_caja_ojo, 2)
            cv2.rectangle(frame, (x_ojos_inicio + x1_int, y_ojos_inicio + y1_int), 
                          (x_ojos_inicio + x2_int, y_ojos_inicio + y2_int), (255, 255, 0), 1)
            cv2.putText(frame, f"Pxs:{pixeles_oscuros} ({porcentaje_oscuro}%)", (centro_x, centro_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja_ojo, 1)

        # Unimos la vista del ojo izquierdo y derecho
        linea_separadora = np.ones((150, 4), dtype=np.uint8) * 255
        ojo_binario_ampliado = np.hstack((vista_ojo_izq, linea_separadora, vista_ojo_der))

        # --- LÓGICA DE SEÑAS (31 Y CIEGO) ---
        izq_abierto = porcentaje_izq >= cfg.UMBRAL_PUPILA
        der_abierto = porcentaje_der >= cfg.UMBRAL_PUPILA

        if izq_abierto and der_abierto:
            self.contador_31 = max(0, self.contador_31 - 3)
            self.contador_ciego = max(0, self.contador_ciego - 3)
        elif not izq_abierto and not der_abierto:
            self.contador_ciego = min(cfg.MAX_CONTADOR, self.contador_ciego + 2)
            self.contador_31 = max(0, self.contador_31 - 3)
        else:
            self.contador_31 = min(cfg.MAX_CONTADOR, self.contador_31 + 2)
            self.contador_ciego = max(0, self.contador_ciego - 2)

        if self.contador_ciego >= cfg.UMBRAL_FRAMES:
            self.textos_senas_activas.append("Sena ciego detectada")
        elif self.contador_31 >= cfg.UMBRAL_FRAMES:
            self.textos_senas_activas.append("Sena 31 detectada")
            
        return ojo_binario_ampliado

    def analizar_lengua(self, frame, x, y, w, h):
        """ ROI de la BOCA Y DETECCIÓN DE 'DOS PITOS' (SACAR LA LENGUA) """
        x_boca_inicio = int(x + w * 0.25)
        x_boca_fin = int(x + w * 0.75)
        y_boca_inicio = int(y + h * 0.65)
        y_boca_fin = int(y + h * 0.95) # Cortamos en 0.95 para no meter barbilla/cuello
        
        cv2.rectangle(frame, (x_boca_inicio, y_boca_inicio), (x_boca_fin, y_boca_fin), (0, 0, 255), 2)
        
        # 1. Recortamos la zona de la boca de la imagen ORIGINAL A COLOR
        roi_boca_color = frame[y_boca_inicio:y_boca_fin, x_boca_inicio:x_boca_fin]
        boca_binaria_ampliada = None
        porcentaje_lengua = 0

        if roi_boca_color.size > 0:
            # 2. Convertimos el recorte de BGR al espacio de color HSV
            hsv_boca = cv2.cvtColor(roi_boca_color, cv2.COLOR_BGR2HSV)

            # 3. Definimos los rangos de color HSV (buscamos tonos rosados/rojizos)
            mascara1 = cv2.inRange(hsv_boca, cfg.BAJO_ROJO_1, cfg.ALTO_ROJO_1)
            mascara2 = cv2.inRange(hsv_boca, cfg.BAJO_ROJO_2, cfg.ALTO_ROJO_2)
            mascara_lengua = cv2.bitwise_or(mascara1, mascara2)

            # 4. EL TRUCO CLÁSICO: APERTURA (Borra líneas finas como los labios cerrados)
            kernel_apertura = np.ones((5, 5), np.uint8)
            mascara_lengua = cv2.morphologyEx(mascara_lengua, cv2.MORPH_OPEN, kernel_apertura)

            # 5. ENCONTRAR CONTORNOS (BLOBS / FORMAS) para no contar píxeles sueltos de ruido
            contornos, _ = cv2.findContours(mascara_lengua, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contornos:
                # Nos quedamos EXCLUSIVAMENTE con la mancha más grande
                contorno_mas_grande = max(contornos, key=cv2.contourArea)
                area_lengua = cv2.contourArea(contorno_mas_grande)
                
                total_pixeles_boca = mascara_lengua.shape[0] * mascara_lengua.shape[1]
                porcentaje_lengua = int((area_lengua / total_pixeles_boca) * 100)

                # Limpiamos la máscara a negro y dibujamos SOLO nuestra mancha ganadora
                mascara_lengua = np.zeros_like(mascara_lengua)
                cv2.drawContours(mascara_lengua, [contorno_mas_grande], -1, 255, cv2.FILLED)
            
            cv2.putText(frame, f"Boca ({porcentaje_lengua}%)", (x_boca_inicio, y_boca_inicio - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            boca_binaria_ampliada = cv2.resize(mascara_lengua, (200, 120))

            # 7. Máquina de estados para Dos Pitos (Lengua)
            if porcentaje_lengua >= cfg.UMBRAL_LENGUA:
                self.contador_lengua = min(cfg.MAX_CONTADOR, self.contador_lengua + 2)
            else:
                self.contador_lengua = max(0, self.contador_lengua - 2)

            if self.contador_lengua >= cfg.UMBRAL_FRAMES:
                self.textos_senas_activas.append("Sena dos pitos (lengua) detectada")

        return boca_binaria_ampliada