"""
======================================================================
LÓGICA DEL JUEGO DE MUS
======================================================================
Este módulo aísla completamente las reglas del juego de la visión artificial.
Recibe los números brutos y devuelve textos formateados para el HUD (interfaz).
"""
from collections import Counter

def evaluar_mus(mano_cruda):
    """
    Recibe una lista de enteros (la mano detectada).
    Devuelve una lista de 4 strings: [Pares, Juego, Grandes, Chicas].
    """
    # Regla 0: Un jugador de mus necesita exactamente 4 cartas.
    # Si la visión no ha detectado las 4, devolvemos una lista vacía.
    if len(mano_cruda) < 4:
        return [] 
    
    # Nos aseguramos de trabajar solo con las 4 primeras por seguridad
    mano_cruda = mano_cruda[:4] 

    # =========================================================
    # REGLA 1: CERDOS Y PITOS
    # El 3 vale por un Rey (12) y el 2 vale por un As (1).
    # Transformamos la mano para aplicar la matemática fácilmente.
    # =========================================================
    mano = [12 if carta == 3 else 1 if carta == 2 else carta for carta in mano_cruda]

    # Diccionario visual para traducir el número al nombre de la figura
    nombres = {
        12: "Rey/3", 11: "Caballo", 10: "Sota", 
        7: "Siete", 6: "Seis", 5: "Cinco", 4: "Cuatro", 1: "As/Pito"
    }

    # =========================================================
    # REGLA 2: PARES (Duples, Medias, Par)
    # Contamos cuántas veces se repite cada valor en la mano.
    # =========================================================
    conteos = Counter(mano)
    valores_pares = list(conteos.values())
    
    texto_pares = "Pareja: No"
    if 4 in valores_pares:
        texto_pares = "Pareja: Si | Duples (4 iguales)"
    elif valores_pares.count(2) == 2:
        # Si hay dos grupos de 2 cartas iguales (ej: dos reyes y dos cuatros)
        texto_pares = "Pareja: Si | Duples (2 pares)"
    elif 3 in valores_pares:
        texto_pares = "Pareja: Si | Medias"
    elif 2 in valores_pares:
        texto_pares = "Pareja: Si | Par"

    # =========================================================
    # REGLA 3: JUEGO Y PUNTO
    # Las figuras valen 10, las blancas su valor nominal.
    # Juego si se alcanza 31.
    # =========================================================
    suma_puntos = sum([10 if c in [10, 11, 12] else c for c in mano])
            
    if suma_puntos >= 31:
        texto_juego = f"Juego: Si | Puntos: {suma_puntos}"
        if suma_puntos == 31:
            texto_juego += " (Max!)"
    else:
        texto_juego = f"Juego: No | Punto: {suma_puntos}"

    # =========================================================
    # REGLA 4: GRANDES Y CHICAS
    # Ordenamos la mano de mayor a menor para sacar los extremos.
    # =========================================================
    mano_ordenada = sorted(mano, reverse=True)
    
    # Las dos mejores (índices 0 y 1 del array ordenado de mayor a menor)
    g1 = nombres.get(mano_ordenada[0], str(mano_ordenada[0]))
    g2 = nombres.get(mano_ordenada[1], str(mano_ordenada[1]))
    
    # Las dos peores (índices -1 y -2, es decir, el final del array)
    c1 = nombres.get(mano_ordenada[-1], str(mano_ordenada[-1]))
    c2 = nombres.get(mano_ordenada[-2], str(mano_ordenada[-2]))

    texto_grandes = f"Grandes: {g1}, {g2}"
    texto_chicas  = f"Chicas: {c1}, {c2}"

    # Retornamos el bloque de texto empaquetado para la interfaz gráfica
    return [texto_pares, texto_juego, texto_grandes, texto_chicas]