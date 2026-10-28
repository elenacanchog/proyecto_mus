import cv2
import numpy as np
import urllib.request
import time

URL = "http://192.168.1.132:8080/shot.jpg"   # tu IP, terminada en /shot.jpg


def capturar():
    """Pide UNA foto al móvil. Devuelve la imagen o None si falla."""
    try:
        with urllib.request.urlopen(URL, timeout=3) as r:
            datos = np.frombuffer(r.read(), dtype=np.uint8)
        return cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error al capturar:", e)
        return None


print("Vista previa lenta (unos 5 fps). ESPACIO = guardar foto | q = salir")
while True:
    t0 = time.time()
    frame = capturar()
    if frame is None:
        time.sleep(1)                       # si falla, esperamos antes de reintentar
        continue

    ms = (time.time() - t0) * 1000
    cv2.putText(frame, f"{ms:.0f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camara", frame)

    tecla = cv2.waitKey(150) & 0xFF         # pausa de 150 ms: no consume CPU
    if tecla == ord('q'):
        break
    if tecla == ord(' '):
        cv2.imwrite("prueba_mesa.jpg", frame)
        print("Foto guardada en prueba_mesa.jpg")

cv2.destroyAllWindows()