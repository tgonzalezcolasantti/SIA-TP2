import argparse
from pathlib import Path
import random

from PIL import Image as pimg
import numpy as np

from gen.cross import OnePointCross, RingCross, TwoPointCross, UniformCross
from gen.genetic import GeneticAlgorithm, RecombinationType
from gen.population import MutationType
from gen.selection import (
    BoltzmannSelection,
    DeterministicTournamentSelection,
    EliteSelection,
    ProbabilisticTournamentSelection,
    RankingSelection,
    RouletteSelection,
    UniversalSelection,
)
from img.image import ImageIndividualFactory, TargetImage, Triangle


def existing_image(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Image file does not exist: {path}")
    return path


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Value must be an integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Value must be a number") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Value must be a number") from error
    if number < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative")
    return number


def probability(value: str) -> float:
    number = nonnegative_float(value)
    if number > 1:
        raise argparse.ArgumentTypeError("Probability must be between 0 and 1")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximate an image with triangles using a genetic algorithm."
    )
    parser.add_argument("image", type=existing_image, help="Path to the target image")
    parser.add_argument(
        "triangle_count",
        type=positive_integer,
        help="Number of triangles used by each individual",
    )
    parser.add_argument("--population-size", type=positive_integer, default=20)
    parser.add_argument("--max-generations", type=positive_integer, default=20000)
    parser.add_argument(
        "--target-error",
        type=nonnegative_float,
        default=0.0,
        help="Stop when the best MSE is at or below this value",
    )
    parser.add_argument(
        "--selection",
        choices=(
            "elite",
            "roulette",
            "universal",
            "boltzmann",
            "tournament-deterministic",
            "tournament-probabilistic",
            "ranking",
        ),
        default="roulette",
    )
    parser.add_argument(
        "--crossover",
        choices=("one-point", "two-point", "ring", "uniform"),
        default="uniform",
    )
    parser.add_argument(
        "--mutation",
        choices=("single", "limited", "uniform", "complete"),
        default="single",
    )
    parser.add_argument("--mutation-probability", type=probability, default=0.2)
    parser.add_argument(
        "--mutation-limit",
        type=positive_integer,
        default=1,
        help="Maximum genes selected by limited mutation",
    )
    parser.add_argument(
        "--survival",
        choices=("additive", "exclusive"),
        default="additive",
    )
    parser.add_argument("--tournament-size", type=positive_integer, default=3)
    parser.add_argument("--tournament-threshold", type=probability, default=0.75)
    parser.add_argument(
        "--boltzmann-initial-temperature",
        type=positive_float,
        default=1000.0,
    )
    parser.add_argument(
        "--boltzmann-target-temperature",
        type=positive_float,
        default=10.0,
    )
    parser.add_argument("--boltzmann-rate", type=nonnegative_float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable the live image preview",
    )
    args = parser.parse_args()
    if not 0.5 <= args.tournament_threshold <= 1:
        parser.error("--tournament-threshold must be between 0.5 and 1")
    if args.boltzmann_target_temperature > args.boltzmann_initial_temperature:
        parser.error(
            "--boltzmann-target-temperature cannot exceed "
            "--boltzmann-initial-temperature"
        )
    if (
        args.mutation == "limited"
        and args.mutation_limit > args.triangle_count * Triangle.genome_length
    ):
        parser.error("--mutation-limit cannot exceed the individual's gene count")
    return args


def create_selection(args: argparse.Namespace):
    if args.selection == "elite":
        return EliteSelection()
    if args.selection == "roulette":
        return RouletteSelection()
    if args.selection == "universal":
        return UniversalSelection()
    if args.selection == "boltzmann":
        return BoltzmannSelection(
            initial_temp=args.boltzmann_initial_temperature,
            target_temp=args.boltzmann_target_temperature,
            rate=args.boltzmann_rate,
        )
    if args.selection == "tournament-deterministic":
        return DeterministicTournamentSelection(args.tournament_size)
    if args.selection == "tournament-probabilistic":
        return ProbabilisticTournamentSelection(args.tournament_threshold)
    return RankingSelection()


def create_crossover(name: str):
    crossovers = {
        "one-point": OnePointCross,
        "two-point": TwoPointCross,
        "ring": RingCross,
        "uniform": UniformCross,
    }
    return crossovers[name]()


def create_mutation(name: str) -> MutationType:
    mutations = {
        "single": MutationType.SINGLEGENE,
        "limited": MutationType.MULTI_LIMITED,
        "uniform": MutationType.MULTI_UNIFORM,
        "complete": MutationType.MULTI_COMPLETE,
    }
    return mutations[name]


def create_survival(name: str) -> RecombinationType:
    survivals = {
        "additive": RecombinationType.ADDITIVE,
        "exclusive": RecombinationType.EXCLUSIVE,
    }
    return survivals[name]


def main():
    args = parse_args()
    random.seed(args.seed)
    with pimg.open(args.image) as image:
        target = np.array(image.convert("RGB"))
    problem = TargetImage(target)
    algo = GeneticAlgorithm(
        target=problem,
        initial_size=args.population_size,
        individual_factory=ImageIndividualFactory(
            target=problem,
            shape=Triangle,
            shape_count=args.triangle_count,
        ),
        selection_method=create_selection(args),
        cross_method=create_crossover(args.crossover),
        mutation_method=create_mutation(args.mutation),
        mutation_probability=args.mutation_probability,
        recombination_method=create_survival(args.survival),
        mutation_multi_limit=args.mutation_limit,
    )
    ans, score = algo.run(
        max_generations=args.max_generations,
        target_score=-args.target_error,
        plot=not args.no_plot,
    )
    print(f"Final MSE: {-score:.6f}")
    pimg.fromarray(ans.render()).save("best_result.png")
    print("Best image saved to best_result.png")
    for triangle in ans.triangles:
        print(triangle.to_svg_polygon(target.shape[1], target.shape[0]))

if __name__ == "__main__":
    main()
