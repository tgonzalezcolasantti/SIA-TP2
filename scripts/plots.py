import csv
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]

colors = ["lightcoral", "red", "sandybrown", "darkgoldenrod", "yellowgreen", "chartreuse", "deepskyblue", "royalblue", "mediumpurple", "hotpink"]

def bar_graph(output_file, data: Mapping[str, Mapping[str, float | int | List[float]]], xlabel: str, ylabel: str):
    bar_width = 1 / (len(data) + 2)
    _, ax = plt.subplots(figsize =(20, 8))
    levels = list(data[list(data.keys())[0]].keys())

    for idx, algo in enumerate(sorted(data)):
        bars_x = np.arange(len(data[algo])) + bar_width * idx
        algo_data = [data[algo][level] for level in levels]
        if isinstance(algo_data[0], List):
            means = [np.mean(x) for x in algo_data] # type: ignore
            errors = [np.std(x) for x in algo_data] # type: ignore
            b = plt.bar(bars_x, means, width=bar_width, color = colors[idx],
                         edgecolor='grey', label=algo, yerr=errors)
            plt.bar_label(b, [f'{x:.4f}' for x in means], padding=5, rotation=90)
        else:
            b = plt.bar(bars_x, algo_data, color=colors[idx], width = bar_width,
                    edgecolor ='grey', label=algo)
            plt.bar_label(b, algo_data, padding=5, rotation=90)

    plt.xlabel(xlabel, fontweight ='bold', fontsize = 15)
    plt.ylabel(ylabel, fontweight ='bold', fontsize = 15)
    plt.xticks([r + 0.5-bar_width for r in range(len(data[list(data.keys())[0]]))], levels)
    ax.set_yscale('log')
    ax.margins(y=0.2)
    plt.legend()
    #plt.show()
    plt.savefig(output_file)

def parse_csv(path: Path, x_main: List[str], x_groupby: str | None, y: str) -> Dict[str, Dict[str, List[float]]]:
    mapping: Dict[str, Dict[str, List[float]]] = {}

    with open(path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for line in reader:
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
                mapping[algokey][" "].append(line[y]) # type: ignore
    return mapping

def main():
    plot_folder = ROOT_DIR / 'results' / 'plots'
    plot_folder.mkdir(parents=True, exist_ok=True)

    time_vs_triangles_map = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['triangles'], 'target', 'time')
    bar_graph(plot_folder / 'time_vs_triangles.png', time_vs_triangles_map, "Tiempo de ejecucion por algoritmo por nivel", "Tiempo")

    time_vs_population_map = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['population_size'], 'target', 'time')
    bar_graph(plot_folder / 'time_vs_population.png', time_vs_population_map, "Tiempo de ejecucion por algoritmo por nivel", "Tiempo")

    precision_vs_population_map = parse_csv(ROOT_DIR / 'results' / 'results.csv', ['population_size'], 'triangles', 'MSE')
    bar_graph(plot_folder / 'precision_triangles_vs_population.png', precision_vs_population_map, "Tiempo de ejecucion por algoritmo por nivel", "Tiempo")
    # bar_graph(plot_folder / 'expanded.png', expanded_nodes, "Nodos expandidos por algoritmo por nivel", "Nodos expandidos")
    # bar_graph(plot_folder / 'frontier.png', frontier_nodes, "Nodos de frontera por algoritmo por nivel", "Nodos en frontera")
    # bar_graph(plot_folder / 'cost.png', cost, "Costo de solucion por algoritmo por nivel", "Costo de solucion")

if __name__ == "__main__":
    main()