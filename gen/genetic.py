from enum import Enum
import random
from statistics import mean, stdev
from typing import Generic, Optional, Self, Sequence, Tuple
from matplotlib import pyplot as plt
from gen.cross import CrossMethod
from gen.population import IndividualT, MutationType, TargetT, Population
from gen.selection import SelectionMethod

class RecombinationType(Enum):
    ADDITIVE = "Additive"
    EXCLUSIVE = "Exclusive"

class GeneticAlgorithm(Generic[IndividualT, TargetT]):
    def __init__(
        self: Self,
        target: TargetT,
        initial_size: int,
        individual: type[IndividualT],
        selection_method: SelectionMethod,
        cross_method: CrossMethod,
        mutation_method: MutationType,
        mutation_probability: float,
        recombination_method: RecombinationType,
        mutation_multi_limit: Optional[int] = None,

    ):
        self.population: Population = Population([individual.from_scratch(target) for _ in range(initial_size)])
        self.target=target
        self.selection_method=selection_method
        self.cross_method=cross_method
        self.mutation_method=mutation_method
        self.mutation_probability=mutation_probability
        self.mutation_multi_limit=mutation_multi_limit
        self.recombination_method=recombination_method

    def run_generation(self: Self) -> None:
        #Step 1: Selection
        best=self.selection_method.select(population=self.population, amount=int(len(self.population.individuals)/2))
        #Step 2: Crossbreeding
        new_population=self.cross_method.cross(best, len(self.population.individuals))
        #Step 3: Mutations
        for individual in new_population.individuals:
            if self.mutation_method == MutationType.SINGLEGENE:
                individual.mutate_single(p=self.mutation_probability)
            elif self.mutation_method == MutationType.MULTI_LIMITED:
                individual.mutate_multi_lim(p=self.mutation_probability, amount=self.mutation_multi_limit)
            elif self.mutation_method == MutationType.MULTI_COMPLETE:
                individual.mutate_complete(p=self.mutation_probability)
            elif self.mutation_method == MutationType.MULTI_UNIFORM:
                individual.mutate_multi_uniform(p=self.mutation_probability)
        #Step 4: Recombine populations
        if self.recombination_method == RecombinationType.ADDITIVE:
            self.population = Population(random.choices(self.population.individuals + new_population.individuals, k=len(self.population.individuals)))
        self.population = new_population #For now population size is constant, so this is valid

    def run(self: Self, max_generations: int = 10000, target_score: float = 0.8) -> Tuple[Sequence[IndividualT], float]:
        plt.ion()
        graph = plt.imshow(self.target.shapes_to_image(self.population.individuals))
        scores = [self.target.total_score(self.population.individuals)]
        for i in range(1, max_generations):
            self.run_generation()
            score = self.target.total_score(self.population.individuals)
            scores.append(score)
            print(f"Generation {i} with score {score:.4f} (mean: {mean(scores):.4f}, sdev: {stdev(scores):.4f})")
            for t in self.population.individuals:
                print(t.to_svg_polygon(self.target.image.shape[1], self.target.image.shape[0])) # type: ignore
            graph.set_data(self.target.shapes_to_image(self.population.individuals))
            plt.draw()
            plt.pause(0.01)
            if score >= target_score:
                break
        return (self.population.individuals, self.target.total_score(self.population.individuals))
