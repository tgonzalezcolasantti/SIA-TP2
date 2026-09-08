import argparse
import csv
from itertools import product
import multiprocessing
from multiprocessing.pool import ApplyResult, ThreadPool
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

DEFAULT_SELECTION = [
    "elite",
    "roulette",
    "universal",
    "boltzmann",
    "ranking",
    "tournament-deterministic",
    "tournament-probabilistic",
]
DEFAULT_CROSSBREED = ["one-point", "ring", "uniform"]

DEFAULT_MUTATION = ["single", "uniform"]
DEFAULT_SURVIVAL = ["additive", "exclusive"]
FIELDNAMES = [
    "target",
    "selection",
    "crossbreed",
    "mutation",
    "prob",
    "survival",
    "generations",
    "MSE",
    "triangles",
    "population_size",
    "time",
]


def parse_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

def benchmark_cases(
    targets: List[str],
    triangles: List[str],
    selection: List[str],
    crossbreed: List[str],
    mutation: List[str],
    mutation_probs: List[str],
    survival: List[str],
    population: List[str],
    seeds: List[str],
) -> List[Tuple[str, str, Dict[str, str]]]:
    cases = [
        (f"{tar} {t} --selection {sel} --crossbreed {cr} --mutation {mut} --no-plot "+\
        f"--mutation-probability {p} --survival {surv} --population-size {pop} --seed {s}",
        f"{Path(tar).name} {t} {sel[0:2]} {cr[0:2]} {mut[0:2]} {p} {surv[0]} {pop} {s}",
        {"target": tar, "triangles": t, "selection": sel, "crossbreed": cr, "mutation": mut,
         "prob": p, "survival": surv, "population_size": pop})
        for tar, t, sel, cr, mut, p, surv, pop, s in product(
            targets,
            triangles,
            selection,
            crossbreed,
            mutation,
            mutation_probs,
            survival,
            population,
            seeds,
        )
    ]
    return cases

def run_task(params: str, row: Dict[str, str], task: TaskID, progress: Progress, timeout: int) -> Dict[str, str]:
    cmd: List[str] = ["uv", "run", "main.py", *params.split()]

    if progress:
        progress.start_task(task)
        progress.update(task, visible=True)
    gen = 0
    with subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1) as proc:
        start = time.time()
        while True:
            result = proc.poll()
            if time.time() - start > timeout:
                row["MSE"] = "N/A"
                row["generations"] = str(gen)
                row["time"] = str(timeout)
            if result is not None:
                break
            if proc.stdout:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    if "Generation" in line and progress is not None:
                        gen = int(line.split()[1])
                        progress.update(task, completed=gen, refresh=True)
                    elif "Final MSE" in line:
                        row["MSE"] = f"{float(line.split(":")[1]):.4f}"
                        row["time"] = f"{(time.time() - start):.4f}"
                        row["generations"] = str(gen)
                        progress.remove_task(task)
                        return row
                    time.sleep(0)

        if progress:
            progress.remove_task(task)
        if result != 0:
            print(f"  ERROR: {params}")
            if proc.stderr is not None:
                print(proc.stderr.readlines())
        row["MSE"] = "N/A"
        row["generations"] = str(gen)
        row["time"] = "N/A"
        return row

def run_simulations(
    targets: List[str],
    selection: List[str],
    crossbreed: List[str],
    mutation: List[str],
    survival: List[str],
    seeds: List[str],
    mutation_probs: list[str],
    tasks: Optional[int],
    timeout: int,
    output: Path,
    triangles: List[str],
    population: List[str],
    writer: csv.DictWriter,
):
    taskprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )
    globalprogress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    table = Table(box=None)
    table.add_row(taskprogress)
    table.add_row(globalprogress)
    rows = []
    with (
        ThreadPool(processes=tasks or int(multiprocessing.cpu_count())) as executor,
        Live(table, refresh_per_second=10),
    ):
        jobs: List[ApplyResult] = []
        for params, title, row in benchmark_cases(
            targets,
            triangles,
            selection,
            crossbreed,
            mutation,
            mutation_probs,
            survival,
            population,
            seeds,
        ):
            task = taskprogress.add_task(
                title,
                start=False,
                total=2000,
                visible=False,
                is_task=True,
            )
            params += f" --output {output / (title.replace(" ", "_") + ".png")}"
            jobs.append(
                executor.apply_async(
                    run_task, (params, row, task, taskprogress, timeout)
                )
            )
        full_progress = globalprogress.add_task(
            f"Total progress ({len(jobs)} elements)", total=len(jobs), is_task=False
        )
        while len(jobs) > 0:
            for job in list(jobs):
                if job.ready():
                    jobs.remove(job)
                    row = job.get()
                    globalprogress.advance(full_progress)
                    rows.append(row)
                    print_row(row)
                    writer.writerow(row)
    return rows

def print_row(row: Dict[str, object]) -> None:
    mse = row["MSE"]
    generations = row["generations"]
    time_sec = row["time"]
    target = row["target"]
    print(f"{target}: {time_sec}, {generations}, {mse}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run comparative triangle image approximators using genetic algorithms"
    )
    parser.add_argument(
        "--images",
        default="test_images/flag_argentina.png,test_images/emoji.png",
        help="Comma-separated target image files",
    )
    parser.add_argument(
        "--triangles",
        default="5,10,20",
        help="Comma-separated triangle count values.",
    )
    parser.add_argument(
        "--population-size",
        default="10,20,30",
        help="Comma-separated population size values.",
    )
    parser.add_argument(
        "--seeds",
        default="1234",
        help="Comma-separated seed values. Will run one run per seed per setting combo.",
    )
    parser.add_argument(
        "--selection",
        default=",".join(DEFAULT_SELECTION),
        help="Comma-separated selection algorithms",
    )
    parser.add_argument(
        "--crossbreed",
        default=",".join(DEFAULT_CROSSBREED),
        help="Comma-separated crossbreed algorithms",
    )
    parser.add_argument(
        "--mutation",
        default=",".join(DEFAULT_MUTATION),
        help="Comma-separated mutation algorithms",
    )
    parser.add_argument(
        "--mutation-probabilities",
        default="0.01,0.1,0.5",
        help="Comma-separated seed values. Will run one run per seed per setting combo.",
    )
    parser.add_argument(
        "--survival",
        default=",".join(DEFAULT_SURVIVAL),
        help="Comma-separated heuristics for greedy/astar",
    )
    parser.add_argument(
        "--timeout", type=float, default=1200.0, help="Timeout per run in seconds"
    )
    parser.add_argument(
        "--output",
        default="results",
        help="output folder where to dump csvs, videos and such",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        help="How many parallel tasks to run.",
    )
    args = parser.parse_args()

    targets = parse_csv_arg(args.images)
    triangles = parse_csv_arg(args.triangles)
    population = parse_csv_arg(args.population_size)
    selection = parse_csv_arg(args.selection)
    crossbreed = parse_csv_arg(args.crossbreed)
    mutation = parse_csv_arg(args.mutation)
    survival = parse_csv_arg(args.survival)
    seeds = parse_csv_arg(args.seeds)
    mutation_probs = parse_csv_arg(args.mutation_probabilities)

    output = ROOT_DIR / str(args.output)
    csvout = output / "results.csv"
    output.mkdir(parents=True, exist_ok=True)
    with csvout.open("w", newline="", encoding="utf-8", buffering=1) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        run_simulations(
            targets=targets,
            selection=selection,
            crossbreed=crossbreed,
            mutation=mutation,
            survival=survival,
            seeds=seeds,
            mutation_probs=mutation_probs,
            tasks=args.tasks,
            timeout=args.timeout,
            output=output,
            triangles=triangles,
            population=population,
            writer=writer,
        )
    print(f"\nSaved benchmark results to {output}")


if __name__ == "__main__":
    main()
