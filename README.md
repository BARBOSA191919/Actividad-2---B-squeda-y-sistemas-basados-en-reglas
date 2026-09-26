# Sistema inteligente de rutas — Transporte público de Neiva

**Actividad 3 — Sistemas inteligentes basados en conocimiento**
Reglas lógicas + búsqueda heurística A\*

## Descripción

Sistema que encuentra la ruta de menor tiempo entre dos paradas del
transporte público de Neiva. Combina dos técnicas:

1. **Base de conocimiento** — hechos y reglas lógicas (estilo Prolog) que
   describen las paradas, las rutas y cuándo hay transbordo entre ellas.
2. **Búsqueda A\*** — con ese conocimiento ya procesado, encuentra el camino
   más rápido, penalizando cada transbordo.

El conocimiento y el algoritmo están separados: para usar otra ciudad u
otras rutas, solo se cambian los datos en `BASE_DE_CONOCIMIENTO`, sin tocar
el resto del código.

## Cómo funciona

- **Motor de inferencia:** aplica las reglas sobre los hechos (encadenamiento
  hacia adelante) hasta deducir todo el conocimiento posible — por ejemplo,
  en qué paradas se puede hacer transbordo, sin tener que escribirlo a mano.
- **Búsqueda A\*:** cada estado es `(parada, ruta actual)`. El costo es el
  tiempo de viaje más la penalización de transbordo. La heurística es la
  distancia en línea recta dividida por la velocidad promedio — nunca
  sobreestima, así que A\* siempre encuentra la ruta óptima.

## Datos

El ejemplo usa **6 lugares reales de Neiva**, con coordenadas verificadas en
OpenStreetMap: Terminal de Transporte, Aeropuerto Benito Salas, Parque
Santander, Plaza de San Pedro, Estadio Guillermo Plazas Alcid y Universidad
Surcolombiana.

> Las paradas son reales, pero las 3 rutas que las conectan (`ruta_1`,
> `ruta_2`, `ruta_3`) son un recorrido ilustrativo: no fue posible acceder
> a los recorridos oficiales del SETP-MIGO (`setpmigo.com`) al hacer este
> trabajo.

## Archivos

- `sistema_rutas.py` — el sistema completo (base de conocimiento, motor de
  inferencia y A\*) en un solo archivo
- `test_sistema_rutas.py` — 8 pruebas automáticas
- `Actividad3_Rutas_Neiva.ipynb` — notebook para Google Colab

## Cómo ejecutar

**Google Colab:** subir el `.ipynb` y usar Entorno de ejecución → Ejecutar
todo.

**Terminal (Python 3.9+, sin dependencias):**

```bash
python sistema_rutas.py aeropuerto_benito_salas universidad_surcolombiana
python sistema_rutas.py aeropuerto_benito_salas universidad_surcolombiana --reglas
python sistema_rutas.py aeropuerto_benito_salas universidad_surcolombiana --sin-heuristica
python -m unittest test_sistema_rutas -v
```

Paradas disponibles: `terminal_transporte`, `aeropuerto_benito_salas`,
`parque_santander`, `san_pedro_plaza`, `estadio_plazas_alcid`,
`universidad_surcolombiana`.

### Ejemplo

```
Ruta de Aeropuerto Benito Salas a Universidad Surcolombiana
-------------------------------------------------------
  Salida: Aeropuerto Benito Salas
  [Toma la Ruta 3] -> Estadio Plazas Alcid
  [Toma la Ruta 1] -> Parque Santander
  [Toma la Ruta 2] -> Universidad Surcolombiana
-------------------------------------------------------
Tiempo total : 39.0 minutos
Transbordos  : 2
Nodos explorados por A*: 5
```

Con búsqueda sin heurística el resultado es el mismo (39.0 min) pero
explorando 7 nodos en vez de 5.

## Adaptar a datos reales

Editar solo los hechos `coord` y `tramo` dentro de `BASE_DE_CONOCIMIENTO`,
en `sistema_rutas.py`. El resto del código no cambia.

## Referencias

- Benítez, R. (2014). *Inteligencia artificial avanzada*. Editorial UOC.
- Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern
  Approach* (4.ª ed.). Pearson.
- OpenStreetMap contributors. https://www.openstreetmap.org
- Sistema Estratégico de Transporte Público de Neiva (SETP-MIGO).
  https://www.setpmigo.com

## Autores

*Santiago José Babosa Rivas*
