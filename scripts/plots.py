import csv
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]

colors = ["lightcoral", "red", "sandybrown", "darkgoldenrod", "yellowgreen", "chartreuse", "deepskyblue", "royalblue", "mediumpurple", "hotpink"]

def bar_graph(output_file, data: Mapping[str, Mapping[str, List[float]]], xlabel: str, ylabel: str, sorter: Optional[Callable[[str], float]] = None):
    bar_width = 1 / (len(data) + 2)
    _, ax = plt.subplots(figsize =(20, 8))
    groups = list(data[list(data.keys())[0]].keys())

    for idx, x in enumerate(sorted(data.keys(), key=sorter)):
        bars_x = np.arange(len(data[x])) + bar_width * idx
        y_data = [data[x][group] for group in groups]
        # if isinstance(y_data[0], List):
        means = [np.mean(i) for i in y_data] # type: ignore
        errors = [np.std(i) for i in y_data] # type: ignore
        b = plt.bar(bars_x, means, width=bar_width, color = colors[idx],
                        edgecolor='grey', label=x, yerr=errors)
        plt.bar_label(b, [f'{i:.4f}' for i in means], padding=5, rotation=90)
        # else:
        #     b = plt.bar(bars_x, y_data, color=colors[idx], width = bar_width,
        #             edgecolor ='grey', label=x)
        #     plt.bar_label(b, y_data, padding=5, rotation=90)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.5-bar_width for r in range(len(data[list(data.keys())[0]]))], groups)
    # ax.set_yscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    plt.savefig(output_file)

def box_plot(output_file: Path, data: Mapping[str, Mapping[str, List[float]]], xlabel: str, ylabel: str):
    _, ax = plt.subplots(figsize =(20, 8))
    groups = list(data[list(data.keys())[0]].keys())
    mydata = []
    for idx, x in enumerate(sorted(data.keys())):
        y_data = []
        for group in groups:
            y_data.extend(data[x][group])
        # if isinstance(y_data[0], List):
        mydata.append(y_data)
        # else:
        #     b = plt.bar(bars_x, y_data, color=colors[idx], width = bar_width,
        #             edgecolor ='grey', label=x)
        #     plt.bar_label(b, y_data, padding=5, rotation=90)
    print(mydata)
    plt.boxplot(mydata)
    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 1 for r in range(len(data))], sorted(data.keys()))
    # ax.set_yscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    plt.savefig(output_file)

def parse_csv(path: Path, x_main: List[str], x_groupby: str | None, y: str, filter: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, List[float]]]:
    mapping: Dict[str, Dict[str, List[float]]] = {}

    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
            if filter is not None:
                valid = True
                for condition in filter.items():
                    if str(line[condition[0]]) != condition[1]:
                        print(f"skipping (condition {condition[0]} is {line[condition[0]]} and not {condition[1]})")
                        valid = False
                        break
                if not valid:
                    continue
            algokey = " ".join([line[x] for x in x_main])
            if algokey not in mapping:
                mapping[algokey] = {}
            if x_groupby is not None:
                if line[x_groupby] not in mapping[algokey]:
                    mapping[algokey][str(line[x_groupby])] = [] # type: ignore
                try:
                    ans = float(line[y])
                    mapping[algokey][str(line[x_groupby])].append(ans) # type: ignore
                except ValueError:
                    continue
            else:
                if " " not in mapping[algokey]:
                    mapping[algokey][" "] = [] # type: ignore
                mapping[algokey][" "].append(float(line[y])) # type: ignore
    return mapping

def main():
    plot_folder = ROOT_DIR / 'results' / 'plots'
    plot_folder.mkdir(parents=True, exist_ok=True)

    time_vs_triangles_map = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['triangles'], 'target', 'time')
    bar_graph(plot_folder / 'time_vs_triangles.png', time_vs_triangles_map, "Tiempo de ejecucion vs cantidad de triangulos", "Tiempo")

    time_vs_population_map = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['population_size'], 'target', 'time')
    bar_graph(plot_folder / 'time_vs_population.png', time_vs_population_map, "Tiempo de ejecucion vs tamano de poblacion", "Tiempo")

    precision_vs_population_map = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['population_size'], 'triangles', 'MSE')
    bar_graph(plot_folder / 'precision_triangles_vs_population.png', precision_vs_population_map, "MSE final vs poblacion y triangulos", "MSE", lambda x: float(x))

    precision_vs_selection = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['selection'], 'target', 'MSE')
    bar_graph(plot_folder / 'precision_vs_selection.png', precision_vs_selection, "MSE final vs algoritmo de seleccion y target", "MSE")

    precision_vs_crossbreed = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['triangles'], 'crossbreed', 'MSE')
    bar_graph(plot_folder / 'precision_vs_crossbreed.png', precision_vs_crossbreed, "MSE final vs algoritmo de cruza y triangulos", "MSE", lambda x: float(x))

    precision_vs_mutation = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['mutation', 'prob'], None, 'MSE')
    box_plot(plot_folder / 'precision_vs_mutation.png', precision_vs_mutation, "MSE final vs algoritmo de mutacion y probabilidad", "MSE")

    precision_vs_survival = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['survival'], 'target', 'MSE')
    bar_graph(plot_folder / 'precision_vs_survival.png', precision_vs_survival, "MSE final vs metodo de supervivencia y target", "MSE")

    # bar_graph(plot_folder / 'expanded.png', expanded_nodes, "Nodos expandidos por algoritmo por nivel", "Nodos expandidos")
    # bar_graph(plot_folder / 'frontier.png', frontier_nodes, "Nodos de frontera por algoritmo por nivel", "Nodos en frontera")
    # bar_graph(plot_folder / 'cost.png', cost, "Costo de solucion por algoritmo por nivel", "Costo de solucion")

if __name__ == "__main__":
    main()