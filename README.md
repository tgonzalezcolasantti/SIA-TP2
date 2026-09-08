# SIA - TP2: aproximación de imágenes con algoritmos genéticos

Trabajo Práctico 2 de Sistemas de Inteligencia Artificial (ITBA).

El programa aproxima una imagen objetivo mediante una composición de triángulos semitransparentes sobre un fondo blanco. La posición, el color y la transparencia de los triángulos evolucionan utilizando un algoritmo genético.

## Requisitos

- [uv](https://docs.astral.sh/uv/)
- Python 3.14 (uv lo instala automaticamente si no esta disponible)

Las dependencias estan declaradas en `pyproject.toml` y fijadas en `uv.lock`.

## Instalación

Desde la raiz del repositorio:

```bash
uv sync
```

Para comprobar la instalacion:

```bash
uv run python main.py --help
```

## Ejecucion rapida

Los dos argumentos obligatorios son la ruta de la imagen y la cantidad de triangulos:

```bash
uv run python main.py test_images/emoji.png 20
```

Al finalizar se muestran el MSE final y la ubicacion de la imagen generada. Por defecto, el resultado se guarda como `best_result.png`.

Ejemplo completo y reproducible:

```bash
uv run python main.py test_images/emoji.png 20 --population-size 30 --max-generations 2000 --target-error 500 --selection boltzmann --crossbreed ring --mutation uniform --mutation-probability 0.01 --survival additive --seed 1234 --output results/emoji.png --no-plot
```

Las imagenes mayores a 128 x 128 pixeles se reducen antes de evaluarlas para limitar el costo computacional.

## Parámetros

| Parametro | Descripcion | Valor predeterminado |
|---|---|---:|
| `image` | Ruta de la imagen objetivo. | Obligatorio |
| `triangle_count` | Cantidad de triangulos por individuo. | Obligatorio |
| `--population-size` | Individuos de cada generacion. | `20` |
| `--max-generations` | Cantidad maxima de generaciones. | `2000` |
| `--target-error` | Finaliza cuando el mejor MSE es menor o igual al valor indicado. | `500` |
| `--selection` | Metodo de seleccion de padres. | `roulette` |
| `--crossbreed` | Metodo de cruza. | `uniform` |
| `--mutation` | Metodo de mutacion. | `single` |
| `--mutation-probability` | Probabilidad de mutacion. | `0.2` |
| `--mutation-limit` | Maximo de genes candidatos para mutacion limitada. | `10` |
| `--survival` | Estrategia de supervivencia. | `additive` |
| `--tournament-size` | Participantes del torneo deterministico. | `3` |
| `--tournament-threshold` | Probabilidad de elegir al mejor en el torneo probabilistico. | `0.75` |
| `--boltzmann-initial-temperature` | Temperatura inicial de Boltzmann. | `1000` |
| `--boltzmann-target-temperature` | Temperatura final de Boltzmann. | `10` |
| `--boltzmann-rate` | Velocidad de enfriamiento. | `0.05` |
| `--seed` | Semilla aleatoria para reproducir una ejecucion. | `1234` |
| `--output` | Archivo PNG de salida. | `best_result.png` |
| `--output-frames` | Carpeta para guardar un PNG por generacion. | Desactivado |
| `--no-plot` | Desactiva la previsualizacion en vivo. | Desactivado |

## Operadores disponibles

### Seleccion

- `elite`
- `roulette`
- `universal`
- `boltzmann`
- `ranking`
- `tournament-deterministic`
- `tournament-probabilistic`

### Cruza

- `one-point`: intercambia los genes desde un punto de corte hasta el final.
- `two-point`: intercambia el segmento comprendido entre dos puntos.
- `ring`: intercambia un segmento que puede continuar desde el final al inicio del genoma.
- `uniform`: decide independientemente que genes intercambiar.
- `bad`: operador de control que no intercambia genes.

### Mutacion

- `single`: con probabilidad `p`, muta un unico gen elegido al azar.
- `limited`: elige hasta `--mutation-limit` genes candidatos y aplica `p` a cada uno.
- `uniform`: aplica `p` independientemente a cada gen.
- `complete`: con probabilidad `p`, muta todos los genes del individuo.

### Supervivencia

- `additive`: combina padres e hijos y conserva los mejores `N`. No pierde la mejor solucion encontrada.
- `exclusive`: reemplaza toda la poblacion por los descendientes. Aumenta la renovacion, pero puede perder buenas soluciones.

## Representación y fitness

Un individuo es una imagen candidata formada por una cantidad fija de triangulos. Cada triangulo aporta 10 genes:

```text
(x1, y1, x2, y2, x3, y3, R, G, B, A)
```

Las coordenadas estan normalizadas en `[0, 1]` y los canales RGBA pertenecen a `[0, 255]`. El orden de los triangulos tambien determina el resultado visual, ya que se dibujan sucesivamente sobre un canvas blanco.

La diferencia entre la imagen generada y la imagen objetivo se calcula mediante el error cuadratico medio:

```text
fitness = -MSE
```

El algoritmo maximiza el fitness; por lo tanto, una solucion mejora cuando su fitness se acerca a cero y su MSE disminuye.

## Ciclo del algoritmo genético

En cada generacion:

1. Se seleccionan los padres mediante el metodo configurado.
2. Se generan `N` descendientes mediante cruza.
3. Se mutan los descendientes.
4. Se forma la siguiente poblacion utilizando supervivencia aditiva o exclusiva.
5. Se actualizan el mejor individuo y su MSE.

La ejecucion termina al alcanzar `--target-error` o completar `--max-generations`, lo que ocurra primero.

## Experimentos en paralelo

`scripts/batch.py` ejecuta el producto cartesiano de los valores recibidos. Las corridas se procesan en paralelo y cada una guarda su imagen junto con una fila de metricas en `results.csv`.

Ejemplo: comparar todos los metodos de seleccion, manteniendo fijos los demas parametros. El batch utiliza los valores predeterminados del programa: hasta 2000 generaciones y MSE objetivo 500.

```bash
uv run python scripts/batch.py --images test_images/emoji.png --triangles 20 --population-size 20 --seeds 1234 --selection elite,roulette,universal,boltzmann,ranking,tournament-deterministic,tournament-probabilistic --crossbreed ring --mutation uniform --mutation-probabilities 0.1 --survival additive --tasks 7 --output results/selection
```

Opciones principales del batch:

- Los valores multiples se separan con comas.
- `--tasks` controla la cantidad maxima de corridas paralelas.
- `--timeout` define el umbral de tiempo de cada corrida para el reporte del batch.
- `--output` selecciona la carpeta para las imagenes y el CSV.

Para ejecutar la grilla experimental predeterminada:

```bash
uv run python scripts/batch.py
```

Esta grilla contiene muchas combinaciones y puede tardar varios minutos.

## Frames y video

Para guardar el mejor individuo de cada generacion:

```bash
uv run python main.py test_images/emoji.png 20 --max-generations 500 --output-frames results/frames --no-plot
```

Los frames pueden convertirse en video con:

```bash
uv run python scripts/video.py results/frames results/evolution.mp4 --fps 60
```

## Estructura del proyecto

```text
.
|-- main.py                         # Punto de entrada
|-- cli.py                          # Argumentos y construccion de operadores
|-- gen/
|   |-- genetic.py                  # Ciclo del algoritmo genetico
|   |-- population.py               # Individuos, poblaciones y mutaciones
|   |-- selection.py                # Metodos de seleccion
|   `-- cross.py                    # Metodos de cruza
|-- img/
|   `-- image.py                    # Triangulos, render y MSE
|-- scripts/
|   |-- batch.py                    # Experimentos paralelos
|   |-- plots.py                    # Graficos de resultados
|   `-- video.py                    # Video a partir de frames
`-- test_images/                    # Imagenes de ejemplo
```

## Autores

- Tomás Agustín González Colasanti - 63281
- Santos Galarraga - 62185
