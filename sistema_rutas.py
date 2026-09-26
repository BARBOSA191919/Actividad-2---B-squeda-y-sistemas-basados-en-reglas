"""
==============================================================================
 SISTEMA INTELIGENTE PARA ENCONTRAR LA MEJOR RUTA EN TRANSPORTE PUBLICO
 URBANO DE NEIVA, HUILA
==============================================================================
Actividad 3 - Sistemas inteligentes basados en conocimiento.

El programa tiene 3 partes, todas en este mismo archivo:
    1) BASE DE CONOCIMIENTO: hechos (paradas reales de Neiva, con sus
       coordenadas, y las rutas que las conectan) y reglas lógicas
       (cuándo hay transbordo, etc.) escritas en un mini-lenguaje
       parecido a Prolog.
    2) MOTOR DE INFERENCIA: lee esas reglas y hechos, y deduce hechos
       nuevos que no estaban escritos explicitamente (por ejemplo, en
       que paradas se puede hacer transbordo).
    3) BUSQUEDA A*: con todo el conocimiento ya deducido, busca el
       camino mas rapido entre dos paradas.

Las 6 paradas de este ejemplo son lugares reales de Neiva (Terminal de
Transporte, Aeropuerto Benito Salas, Parque Santander, Plaza de San
Pedro, Estadio Guillermo Plazas Alcid y Universidad Surcolombiana), con
coordenadas verificadas en OpenStreetMap. Las 3 rutas que las conectan
(Ruta 1, Ruta 2, Ruta 3) son un recorrido ilustrativo construido para
esta actividad, ya que el listado oficial de recorridos del SETP-MIGO
no estaba disponible en el momento de hacer este trabajo.
==============================================================================
"""

import re          # para leer (parsear) el texto de la base de conocimiento
import math        # para calcular distancias entre coordenadas
import heapq       # cola de prioridad que usa el algoritmo A*
import argparse    # para poder ejecutar el programa desde la terminal


# ==============================================================================
# PARTE 1: BASE DE CONOCIMIENTO
# ------------------------------------------------------------------------------
# HECHOS (informacion concreta) y REGLAS (conocimiento general que permite
# deducir hechos nuevos), en un mini-lenguaje logico parecido a Prolog.
# Cada linea termina en punto.
#
#   - Los HECHOS se ven asi:           tramo(parada_a, parada_b, ruta_1).
#   - Las REGLAS se ven asi:           cabeza :- condicion1, condicion2.
#     (se lee: "cabeza es verdad SI condicion1 Y condicion2 son verdad")
#   - Las MAYUSCULAS son variables (representan "cualquier cosa").
#   - Las minusculas son valores concretos (constantes).
# ==============================================================================
BASE_DE_CONOCIMIENTO = """
% ---- HECHOS: coord(parada, latitud, longitud) ----
% Coordenadas reales verificadas en OpenStreetMap (Neiva, Huila).
coord(terminal_transporte, 2.91681, -75.28191).
coord(aeropuerto_benito_salas, 2.95166, -75.29405).
coord(parque_santander, 2.92633, -75.28914).
coord(san_pedro_plaza, 2.92317, -75.29045).
coord(estadio_plazas_alcid, 2.93600, -75.28037).
coord(universidad_surcolombiana, 2.94302, -75.30038).

% ---- HECHOS: tramo(parada_1, parada_2, ruta) ----
% Cada ruta se describe como su secuencia de paradas consecutivas.
% Ruta 1: recorrido sur-centro-norte, pasando por el centro de la ciudad.
tramo(terminal_transporte, san_pedro_plaza, ruta_1).
tramo(san_pedro_plaza, parque_santander, ruta_1).
tramo(parque_santander, estadio_plazas_alcid, ruta_1).

% Ruta 2: conecta el centro con la Universidad Surcolombiana.
tramo(parque_santander, universidad_surcolombiana, ruta_2).

% Ruta 3: conecta el norte de la ciudad con el aeropuerto.
tramo(estadio_plazas_alcid, aeropuerto_benito_salas, ruta_3).

% ---- HECHOS: parametros del sistema ----
velocidad_promedio_kmh(15).
minutos_por_parada(1).
penalizacion_transbordo(6).

% ---- REGLAS ----
% R1 y R2: las busetas circulan en ambos sentidos de la via.
conexion(X, Y, R) :- tramo(X, Y, R).
conexion(X, Y, R) :- tramo(Y, X, R).

% R3 y R4: una ruta "para" en cada estacion que aparece en sus tramos.
parada(E, R) :- tramo(E, _, R).
parada(E, R) :- tramo(_, E, R).

% R5: hay transbordo en una estacion E si dos rutas DISTINTAS paran ahi.
transbordo(E, R1, R2) :- parada(E, R1), parada(E, R2), distinta(R1, R2).
"""


# ==============================================================================
# PARTE 2: MOTOR DE INFERENCIA
# ------------------------------------------------------------------------------
# Lee el texto de arriba, lo convierte en estructuras de Python, y aplica las
# reglas una y otra vez hasta que ya no se pueda deducir nada nuevo. A esta
# tecnica se le llama "encadenamiento hacia adelante" (forward chaining): se
# parte de los hechos conocidos y se avanza generando hechos nuevos, en vez
# de partir de una pregunta.
# ==============================================================================

def _es_variable(termino):
    """Una MAYUSCULA inicial (o "_") indica que el termino es una variable."""
    return isinstance(termino, str) and (termino[:1].isupper() or termino.startswith("_"))


def _convertir_valor(texto):
    """Convierte '2.93' a numero; si no es numero, lo deja como texto (constante)."""
    texto = texto.strip()
    try:
        return int(texto)
    except ValueError:
        try:
            return float(texto)
        except ValueError:
            return texto


def _parsear_atomo(texto, contador=[0]):
    """Convierte el texto 'tramo(a, b, ruta_1)' en ('tramo', ('a','b','ruta_1'))."""
    coincide = re.match(r"\s*(\w+)\((.*)\)\s*$", texto)
    nombre = coincide.group(1)
    argumentos = []
    for arg in coincide.group(2).split(","):
        arg = arg.strip()
        if arg == "_":                       # variable anonima (no importa su valor)
            contador[0] += 1
            arg = f"_anon{contador[0]}"
        argumentos.append(_convertir_valor(arg))
    return nombre, tuple(argumentos)


class MotorDeInferencia:
    """Guarda los hechos y las reglas, y sabe deducir hechos nuevos."""

    def __init__(self):
        self.hechos = {}        # ejemplo: {"tramo": {(a,b,r), (b,c,r), ...}}
        self.reglas = []        # lista de (cabeza, cuerpo, texto_original)
        self.hechos_deducidos = 0

    def cargar_texto(self, texto_kb):
        """Lee el texto de la base de conocimiento y separa hechos de reglas."""
        texto_sin_comentarios = re.sub(r"%.*", "", texto_kb)
        for sentencia in re.split(r"\.\s*(?:\n|$)", texto_sin_comentarios):
            sentencia = sentencia.strip()
            if not sentencia:
                continue
            if ":-" in sentencia:                          # es una REGLA
                cabeza_txt, cuerpo_txt = sentencia.split(":-")
                cuerpo = [_parsear_atomo(a) for a in re.findall(r"\w+\([^)]*\)", cuerpo_txt)]
                self.reglas.append((_parsear_atomo(cabeza_txt), cuerpo, " ".join(sentencia.split())))
            else:                                          # es un HECHO
                nombre, args = _parsear_atomo(sentencia)
                self.hechos.setdefault(nombre, set()).add(args)
        return self

    def _unificar(self, patron, datos, sustitucion):
        """Intenta hacer que 'patron' (con variables) coincida con 'datos' (valores)."""
        nueva = dict(sustitucion)
        for p, v in zip(patron, datos):
            if _es_variable(p):
                if p in nueva and nueva[p] != v:
                    return None                            # la variable ya tenia otro valor
                nueva[p] = v
            elif p != v:
                return None                                # la constante no coincide
        return nueva

    def _resolver_cuerpo(self, cuerpo, sustitucion):
        """Recorre las condiciones de una regla y genera todas las soluciones posibles."""
        if not cuerpo:
            yield sustitucion
            return
        nombre, args = cuerpo[0]
        if nombre == "distinta":                           # condicion especial: A != B
            a = sustitucion.get(args[0], args[0])
            b = sustitucion.get(args[1], args[1])
            if a != b:
                yield from self._resolver_cuerpo(cuerpo[1:], sustitucion)
            return
        for dato in list(self.hechos.get(nombre, ())):
            resultado = self._unificar(args, dato, sustitucion)
            if resultado is not None:
                yield from self._resolver_cuerpo(cuerpo[1:], resultado)

    def aplicar_reglas(self):
        """Aplica las reglas repetidamente hasta que no se deduzca nada nuevo (punto fijo)."""
        se_dedujo_algo = True
        while se_dedujo_algo:
            se_dedujo_algo = False
            for (nombre_cabeza, args_cabeza), cuerpo, _texto in self.reglas:
                for sustitucion in list(self._resolver_cuerpo(cuerpo, {})):
                    hecho_nuevo = tuple(
                        sustitucion[a] if _es_variable(a) else a for a in args_cabeza
                    )
                    conjunto = self.hechos.setdefault(nombre_cabeza, set())
                    if hecho_nuevo not in conjunto:
                        conjunto.add(hecho_nuevo)
                        self.hechos_deducidos += 1
                        se_dedujo_algo = True
        return self

    def consultar(self, nombre_predicado):
        """Devuelve todos los hechos conocidos (o deducidos) de un predicado."""
        return self.hechos.get(nombre_predicado, set())

    def existe(self, nombre_predicado, *args):
        """Responde True/False: existe ese hecho exacto?"""
        return tuple(args) in self.hechos.get(nombre_predicado, set())


# ==============================================================================
# PARTE 3: BUSQUEDA HEURISTICA A*
# ------------------------------------------------------------------------------
# Con el conocimiento ya deducido, buscamos el camino de menor tiempo entre
# dos paradas. Un "estado" en esta busqueda no es solo la parada donde
# estamos, sino el par (parada, ruta_en_la_que_vamos), porque asi sabemos
# si al cambiar de buseta hay que sumar la penalizacion de transbordo.
# ==============================================================================

def _distancia_km(coord_1, coord_2):
    """Distancia en linea recta entre dos puntos (formula de Haversine)."""
    (lat1, lon1), (lat2, lon2) = coord_1, coord_2
    radio_tierra = 6371.0
    f1, f2 = math.radians(lat1), math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin((f2 - f1) / 2) ** 2 + math.cos(f1) * math.cos(f2) * math.sin(delta_lon / 2) ** 2
    return 2 * radio_tierra * math.asin(math.sqrt(a))


class Planificador:
    def __init__(self, texto_kb=BASE_DE_CONOCIMIENTO):
        # 1) cargamos y deducimos todo el conocimiento
        self.kb = MotorDeInferencia().cargar_texto(texto_kb).aplicar_reglas()

        # 2) preparamos estructuras rapidas de consulta
        self.coordenadas = {parada: (lat, lon) for parada, lat, lon in self.kb.consultar("coord")}
        obtener_parametro = lambda nombre: next(iter(self.kb.consultar(nombre)))[0]
        self.velocidad_km_min = obtener_parametro("velocidad_promedio_kmh") / 60
        self.minutos_por_parada = obtener_parametro("minutos_por_parada")
        self.penalizacion_transbordo = obtener_parametro("penalizacion_transbordo")

        # 3) armamos la lista de vecinos de cada parada, con el tiempo de viaje
        self.vecinos = {}
        for origen, destino, ruta in self.kb.consultar("conexion"):
            minutos = _distancia_km(self.coordenadas[origen], self.coordenadas[destino]) \
                      / self.velocidad_km_min + self.minutos_por_parada
            self.vecinos.setdefault(origen, []).append((destino, ruta, round(minutos, 2)))

    def _heuristica(self, parada, destino, usar_heuristica):
        """Estimacion optimista del tiempo restante. Nunca sobreestima (ninguna
        buseta viaja mas rapido que en linea recta), por eso A* garantiza
        encontrar la ruta OPTIMA."""
        if not usar_heuristica:
            return 0.0
        return _distancia_km(self.coordenadas[parada], self.coordenadas[destino]) / self.velocidad_km_min

    def buscar_ruta(self, origen, destino, usar_heuristica=True):
        """Algoritmo A*: explora primero los caminos mas prometedores
        (menor costo_real + heuristica)."""
        for parada in (origen, destino):
            if parada not in self.coordenadas:
                raise ValueError(f"Parada desconocida: {parada}")

        estado_inicial = (origen, None)                 # (parada, ruta_actual); None = aun no toma buseta
        frontera = [(self._heuristica(origen, destino, usar_heuristica), 0, estado_inicial)]
        costo_real = {estado_inicial: 0}
        padre = {estado_inicial: None}
        nodos_expandidos = 0

        while frontera:
            _, costo_hasta_aqui, estado = heapq.heappop(frontera)
            if costo_hasta_aqui > costo_real[estado]:
                continue                                  # ya encontramos algo mejor antes
            parada_actual, ruta_actual = estado
            if parada_actual == destino:
                return self._reconstruir_camino(padre, estado, costo_hasta_aqui, nodos_expandidos)
            nodos_expandidos += 1

            for vecino, ruta_del_tramo, minutos_tramo in self.vecinos.get(parada_actual, []):
                costo_extra = 0
                if ruta_actual is not None and ruta_del_tramo != ruta_actual:
                    # cambiar de ruta solo es valido si la regla 'transbordo' lo permite
                    if not self.kb.existe("transbordo", parada_actual, ruta_actual, ruta_del_tramo):
                        continue
                    costo_extra = self.penalizacion_transbordo

                nuevo_estado = (vecino, ruta_del_tramo)
                nuevo_costo = costo_hasta_aqui + minutos_tramo + costo_extra
                if nuevo_costo < costo_real.get(nuevo_estado, math.inf):
                    costo_real[nuevo_estado] = nuevo_costo
                    padre[nuevo_estado] = estado
                    prioridad = nuevo_costo + self._heuristica(vecino, destino, usar_heuristica)
                    heapq.heappush(frontera, (prioridad, nuevo_costo, nuevo_estado))

        return None   # no existe camino

    def _reconstruir_camino(self, padre, estado_final, costo_total, nodos_expandidos):
        """Sigue los punteros 'padre' desde el destino hasta el origen."""
        camino = []
        estado = estado_final
        while estado is not None:
            camino.append(estado)
            estado = padre[estado]
        camino.reverse()

        transbordos = [
            (camino[i - 1][0], camino[i - 1][1], camino[i][1])
            for i in range(2, len(camino))
            if camino[i][1] != camino[i - 1][1]
        ]
        return {
            "camino": camino,
            "minutos": round(costo_total, 1),
            "transbordos": transbordos,
            "nodos_expandidos": nodos_expandidos,
        }


# ==============================================================================
# FUNCIONES PARA MOSTRAR LOS RESULTADOS EN PANTALLA
# ==============================================================================

def _nombre_legible(parada):
    return parada.replace("_", " ").title()


def imprimir_ruta(resultado, origen, destino):
    if resultado is None:
        print("No existe una ruta entre esas dos paradas.")
        return
    print(f"\nRuta de {_nombre_legible(origen)} a {_nombre_legible(destino)}")
    print("-" * 55)
    ruta_anterior = None
    for parada, ruta in resultado["camino"]:
        if ruta is None:
            print(f"  Salida: {_nombre_legible(parada)}")
        else:
            if ruta != ruta_anterior:
                print(f"  [Toma la {_nombre_legible(ruta)}]")
            print(f"    -> {_nombre_legible(parada)}")
            ruta_anterior = ruta
    print("-" * 55)
    print(f"Tiempo total : {resultado['minutos']} minutos")
    print(f"Transbordos  : {len(resultado['transbordos'])}")
    for parada, ruta_1, ruta_2 in resultado["transbordos"]:
        print(f"   - en {_nombre_legible(parada)}: {_nombre_legible(ruta_1)} -> {_nombre_legible(ruta_2)}")
    print(f"Nodos explorados por A*: {resultado['nodos_expandidos']}")


# ==============================================================================
# PROGRAMA PRINCIPAL
# ------------------------------------------------------------------------------
# Se ejecuta cuando el archivo se corre directamente. Desde la terminal:
#     python sistema_rutas.py origen destino
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mejor ruta en transporte publico de Neiva")
    parser.add_argument("origen", nargs="?", default="terminal_transporte")
    parser.add_argument("destino", nargs="?", default="aeropuerto_benito_salas")
    parser.add_argument("--reglas", action="store_true", help="muestra las reglas y hechos deducidos")
    parser.add_argument("--sin-heuristica", action="store_true", help="compara con busqueda sin heuristica")
    argumentos = parser.parse_args()

    planificador = Planificador()

    if argumentos.reglas:
        print(f"{len(planificador.kb.reglas)} reglas cargadas, "
              f"{planificador.kb.hechos_deducidos} hechos deducidos:\n")
        for _, _, texto in planificador.kb.reglas:
            print("  ", texto)

    resultado = planificador.buscar_ruta(
        argumentos.origen.lower(), argumentos.destino.lower(),
        usar_heuristica=not argumentos.sin_heuristica,
    )
    imprimir_ruta(resultado, argumentos.origen.lower(), argumentos.destino.lower())
