"""
======================================================================
ARCHIVO PRINCIPAL (MAIN) - DETECCIÓN DE MUS EN DIRECTO
======================================================================
Este es el punto de entrada del programa. Ejecuta este archivo para jugar.
Se encarga de unir la cámara, la visión artificial y las reglas del Mus,
además de dibujar la interfaz visual (HUD) en la pantalla.
"""
import cv2
import numpy as np
import time
import math
from collections import Counter

# --- IMPORTAMOS NUESTROS MÓDULOS ---
import config_cartas as cfg
import detector_cartas as vision
import logica_mus as mus

def iniciar_mus_directo():
    # =================================================================
    # 1. INICIALIZACIÓN DEL SISTEMA
    # =================================================================
    # Cargamos el cerebro visual (los moldes) en la memoria RAM
    diccionario_moldes = vision.cargar_diccionario_moldes()
    
    # Encendemos la cámara usando los parámetros de config_cartas.py
    cap = cv2.VideoCapture(cfg.CAMARA_INDICE) 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.CAMARA_BUFFER) # Anti-lag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMARA_ANCHO) # Forzar 1080p
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMARA_ALTO)

    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara en el puerto {cfg.CAMARA_INDICE}.")
        print("Revisa config_cartas.py o asegúrate de que Iriun está conectado.")
        return

    # Variables de control de tiempo y memoria espacial
    memoria_mesa = [] 
    tiempo_anterior = time.time()
    
    print("\n[SISTEMA ACTIVO] Pon las 4 cartas en la mesa. (Pulsa 'Q' para salir)\n")

    # =================================================================
    # 2. BUCLE INFINITO DE VÍDEO (El motor en tiempo real)
    # =================================================================
    while True:
        ret, frame = cap.read()
        if not ret: 
            print("Se perdió la señal de vídeo.")
            break

        # --- A. BÚSQUEDA DE SILUETAS EN LA MESA ---
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Suavizado agresivo para matar el ruido de la cámara (el "grano" de la oscuridad)
        suavizado = cv2.GaussianBlur(gris, (15, 15), 0)
        
        # Binarización adaptativa: Separa mágicamente las cartas (claras) de la mesa (oscura)
        binarizada = cv2.adaptiveThreshold(suavizado, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 51, 10)
        
        # Cierre morfológico: "Pega" los bordes rotos por el ruido para que las cartas sean sólidas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        binarizada = cv2.morphologyEx(binarizada, cv2.MORPH_CLOSE, kernel)
        
        # Encontramos todos los contornos (manchas blancas) en la imagen
        contornos, _ = cv2.findContours(binarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cartas_detectadas_en_mesa = []

        # --- B. PROCESAMIENTO DE CADA CARTA DETECTADA ---
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            
            # Filtro 1: ¿Tiene el tamaño de una carta según nuestra configuración?
            if cfg.AREA_MINIMA < area < cfg.AREA_MAXIMA:
                rect = cv2.minAreaRect(contorno)
                (cx, cy), (w_r, h_r), angulo = rect
                
                # Filtro 2: ¿Tiene proporciones de carta? (Evita detectar móviles o manos)
                if w_r == 0 or h_r == 0: continue
                aspect_ratio = float(w_r) / h_r if w_r > h_r else float(h_r) / w_r
                if aspect_ratio > cfg.ASPECT_RATIO_MAX: continue

                esquinas = cv2.boxPoints(rect)
                esquinas = np.intp(esquinas)
                
                # Ordenamos las esquinas para no planchar la carta del revés
                esquinas_ordenadas = vision.ordenar_puntos(esquinas)
                
                # PARCHE ANTI-DEFORMACIÓN: Si la carta está tumbada, giramos el recorte
                # para plancharla de pie y que el número no se estire y se deforme.
                lado_superior = np.linalg.norm(esquinas_ordenadas[0] - esquinas_ordenadas[1])
                lado_izquierdo = np.linalg.norm(esquinas_ordenadas[0] - esquinas_ordenadas[3])
                if lado_superior > lado_izquierdo:
                    esquinas_ordenadas = np.roll(esquinas_ordenadas, 1, axis=0)
                
                # Planchado (Transformación de Perspectiva) a la medida perfecta
                puntos_destino = np.array([[0, 0], [cfg.CARTA_ANCHO-1, 0], 
                                           [cfg.CARTA_ANCHO-1, cfg.CARTA_ALTO-1], 
                                           [0, cfg.CARTA_ALTO-1]], dtype="float32")
                matriz = cv2.getPerspectiveTransform(esquinas_ordenadas, puntos_destino)
                carta_extraida = cv2.warpPerspective(frame, matriz, (cfg.CARTA_ANCHO, cfg.CARTA_ALTO))

                # TRÍPODE DIGITAL: Aplicamos filtro de nitidez (Sharpening) para contrarrestar
                # el desenfoque de la cámara si la estás aguantando a pulso.
                kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                carta_extraida = cv2.filter2D(carta_extraida, -1, kernel_sharpen)

                # --- EXTRACCIÓN Y OCR ---
                # Pasamos la carta limpia al motor de visión para que nos devuelva el número
                molde_vivo, _ = vision.extraer_numero(carta_extraida)
                numero_identificado = None
                
                if molde_vivo is not None:
                    numero_identificado = vision.reconocer_numero(molde_vivo, diccionario_moldes)
                
                # --- MEMORIA ESPACIAL (ANTI-PARPADEO) ---
                # Evita que un 12 baile y se convierta en un 4 por culpa de una sombra puntual.
                numero_estable = None
                carta_encontrada = False
                
                for mem in memoria_mesa:
                    # Comprobamos si esta carta ya estaba en esa zona de la mesa
                    distancia = math.hypot(mem['cx'] - cx, mem['cy'] - cy)
                    if distancia < cfg.TRACKING_DISTANCIA_MAX:
                        carta_encontrada = True
                        mem['cx'], mem['cy'] = cx, cy # Actualizamos su posición exacta
                        mem['vida'] = cfg.TRACKING_VIDA_INICIAL
                        
                        if numero_identificado is not None:
                            mem['historial'].append(numero_identificado)
                            if len(mem['historial']) > cfg.TRACKING_HISTORIAL_MAX:
                                mem['historial'].pop(0)
                                
                        # El número real es la MODA (El que más veces ha visto en los últimos frames)
                        if len(mem['historial']) > 0:
                            numero_estable = Counter(mem['historial']).most_common(1)[0][0]
                        break
                        
                if not carta_encontrada:
                    hist_inicial = [numero_identificado] if numero_identificado else []
                    memoria_mesa.append({'cx': cx, 'cy': cy, 'historial': hist_inicial, 'vida': cfg.TRACKING_VIDA_INICIAL})
                    numero_estable = numero_identificado

                # --- DIBUJADO DE LA CARTA EN LA MESA ---
                if numero_estable is not None:
                    cartas_detectadas_en_mesa.append(numero_estable)
                    
                    # Dibujamos el recuadro verde alrededor de la carta física
                    cv2.drawContours(frame, [esquinas], -1, (0, 255, 0), 3)
                    
                    # Dibujamos el número amarillo brillante flotando sobre la carta
                    posicion_texto = tuple(esquinas_ordenadas[0].astype(int))
                    cv2.putText(frame, str(numero_estable), (posicion_texto[0], posicion_texto[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6) # Sombra negra
                    cv2.putText(frame, str(numero_estable), (posicion_texto[0], posicion_texto[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3) # Texto amarillo

        # Limpieza de memoria (Si quitamos una carta de la mesa, su "vida" se agota y se borra)
        memoria_viva = []
        for mem in memoria_mesa:
            mem['vida'] -= 1
            if mem['vida'] > 0:
                memoria_viva.append(mem)
        memoria_mesa = memoria_viva

        # =================================================================
        # 3. INTERFAZ GRÁFICA (HUD) Y LÓGICA DEL MUS
        # =================================================================
        # Cálculo de los FPS (Rendimiento)
        tiempo_actual = time.time()
        fps = 1 / (tiempo_actual - tiempo_anterior) if (tiempo_actual - tiempo_anterior) > 0 else 0
        tiempo_anterior = tiempo_actual
        
        # Info técnica (Arriba a la izquierda)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Mesa: {cartas_detectadas_en_mesa}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Pasamos las cartas detectadas al módulo de lógica matemática (logica_mus.py)
        if len(cartas_detectadas_en_mesa) >= 4:
            lineas_mus = mus.evaluar_mus(cartas_detectadas_en_mesa)
            
            # Dibujamos las líneas de Grandes, Chicas, Pares y Juego en cascada
            y_inicial = 130
            for i, linea in enumerate(lineas_mus):
                y_offset = y_inicial + (i * 40)
                cv2.putText(frame, linea, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(frame, linea, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            # Alerta roja si falta alguna carta para poder evaluar el mus
            aviso = f"Esperando 4 cartas... (Detectadas: {len(cartas_detectadas_en_mesa)})"
            cv2.putText(frame, aviso, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(frame, aviso, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Mostramos la ventana final
        cv2.imshow('Analisis de MUS en Directo (Modular)', frame)

        # Salir con la tecla 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpieza de recursos al cerrar
    cap.release()
    cv2.destroyAllWindows()

# Lanzador del programa
if __name__ == "__main__":
    iniciar_mus_directo()