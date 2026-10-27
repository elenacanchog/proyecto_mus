# ==============================================================================
# main_senas.py
# Archivo de ejecución. Inicia la cámara y gestiona la visualización.
# ==============================================================================
import cv2
# Importamos nuestra clase desde el archivo detector_senas.py
from detector_senas import DetectorSenasMus

def main():
    # 2. Iniciar la captura de vídeo. El '0' indica la webcam predeterminada.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara.")
        return

    print("Presiona 'q' para salir del programa.")
    
    # Instanciamos el motor lógico de señas
    detector = DetectorSenasMus()

    # 3. Bucle principal de procesamiento fotograma a fotograma
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el fotograma.")
            break
        
        # Modo espejo para que tu ojo izquierdo esté a la izquierda de la pantalla
        frame = cv2.flip(frame, 1)

        # Delegamos TODO el análisis pesado a nuestra clase, y nos devuelve
        # el frame con las cajas, las ventanas de depuración, y una lista de textos.
        frame, ventanas, senas_detectadas = detector.procesar_fotograma(frame)

        # Mostrar los textos de las señas activas ARRIBA A LA IZQUIERDA
        y_texto = 40
        for sena in senas_detectadas:
            color = (0, 0, 255) # Rojo por defecto (31 y Ciego)
            if "duples" in sena.lower():
                color = (255, 0, 180) # Morado para cejas
            elif "pitos" in sena.lower():
                color = (255, 0, 255) # Magenta para la lengua
                
            cv2.putText(frame, sena, (20, y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            y_texto += 40 # Bajamos la Y para que no se superpongan si hay 2 señas a la vez

        # 7. Mostrar el fotograma procesado en múltiples ventanas
        cv2.imshow('1. Escala de Grises (Sin corregir)', ventanas['gray'])
        cv2.imshow('2. Filtro CLAHE (Luz corregida)', ventanas['gray_clahe'])
        cv2.imshow('3. Deteccion de Rostro - Presiona Q para salir', frame)
        
        # 7.5 Mostrar ventanas de depuración de filtros binarios y HSV
        if ventanas['ojos_bin'] is not None:
            cv2.imshow('4. Filtro Blanco y Negro (Ojo Izq | Ojo Der)', ventanas['ojos_bin'])
        if ventanas['boca_bin'] is not None:
            cv2.imshow('5. Filtro HSV (Boca - Lengua)', ventanas['boca_bin'])
        if ventanas['cejas_bin'] is not None:
            cv2.imshow('6. Filtro Binario (Cejas - Duples)', ventanas['cejas_bin'])

        # 8. Condición de salida
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 9. Limpieza
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()