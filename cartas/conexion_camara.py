import cv2
import time

def iniciar_camara():
    # En Ubuntu, la cámara de tu portátil suele ser la 0.
    # Iriun (la cámara virtual) suele asignarse automáticamente a la 1, 2 o 3.
    indice_camara = 2  
    
    print(f"Intentando conectar a la cámara en el puerto {indice_camara}...")
    cap = cv2.VideoCapture(indice_camara)

    # TRUCO ANTI-LAG: Obligamos a OpenCV a no guardar fotogramas viejos en la cola.
    # Solo procesará la imagen más reciente, vital para el tiempo real.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: No se detecta señal en el puerto {indice_camara}.")
        print("Prueba a cambiar 'indice_camara' a 0, 2 o 3.")
        return

    print("¡Conexión establecida con éxito! Pulsa 'Q' en la ventana para cerrar.")
    
    tiempo_anterior = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Se perdió la señal de la cámara.")
            break

        # Calcular los FPS (Fotogramas por segundo) para medir la fluidez real
        tiempo_actual = time.time()
        fps = 1 / (tiempo_actual - tiempo_anterior) if (tiempo_actual - tiempo_anterior) > 0 else 0
        tiempo_anterior = tiempo_actual

        # Dibujar los FPS en verde arriba a la izquierda
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Prueba de Camara (Iriun)', frame)

        # Salir del bucle si se pulsa la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas de forma segura
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_camara()