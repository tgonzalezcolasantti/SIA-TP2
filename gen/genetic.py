from enum import Enum
import random
from typing import Generic, List, Self, Sequence, Tuple

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
        mutation_multi_limit: int,
        recombination_method: RecombinationType,

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
            p = random.random()
            if self.mutation_method == MutationType.SINGLEGENE:
                individual.mutate_single(p=p)
            elif self.mutation_method == MutationType.MULTI_LIMITED:
                individual.mutate_multi_lim(p=p, amount=self.mutation_multi_limit)
            elif self.mutation_method == MutationType.MULTI_COMPLETE:
                individual.mutate_complete(p=p)
            elif self.mutation_method == MutationType.MULTI_UNIFORM:
                individual.mutate_multi_uniform(p=p)
        #Step 4: Recombine populations
        if self.recombination_method == RecombinationType.ADDITIVE:
            self.population = Population(random.choices(self.population.individuals + new_population.individuals, k=len(self.population.individuals)))
        self.population = new_population #For now population size is constant, so this is valid

    def run(self: Self, max_generations: int = 10000, target_score: float = 0.8) -> Tuple[Sequence[IndividualT], float]:
        for i in range(max_generations):
            self.run_generation()
            score = self.target.total_score(self.population.individuals)
            print(f"Generation {i} with score {score}")
            if score >= target_score:
                break
        return (self.population.individuals, self.target.total_score(self.population.individuals))
