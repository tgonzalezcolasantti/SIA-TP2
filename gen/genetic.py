from enum import Enum
from typing import Generic, Optional, Self, Tuple
from matplotlib import pyplot as plt
from gen.cross import CrossMethod
from gen.population import IndividualFactoryT, IndividualT, MutationType, TargetT, Population
from gen.selection import SelectionMethod

class RecombinationType(Enum):
    ADDITIVE = "Additive"
    EXCLUSIVE = "Exclusive"

class GeneticAlgorithm(Generic[IndividualT, TargetT, IndividualFactoryT]):
    def __init__(
        self: Self,
        target: TargetT,
        initial_size: int,
        individual_factory: IndividualFactoryT,
        selection_method: SelectionMethod,
        cross_method: CrossMethod,
        mutation_method: MutationType,
        mutation_probability: float,
        recombination_method: RecombinationType,
        mutation_multi_limit: Optional[int] = None,

    ):
        if initial_size < 1:
            raise ValueError("initial_size must be at least 1")
        if mutation_probability < 0 or mutation_probability > 1:
            raise ValueError("mutation_probability must be between 0 and 1")
        self.population: Population = Population([individual_factory.create() for _ in range(initial_size)])
        self.target=target
        self.selection_method=selection_method
        self.cross_method=cross_method
        self.mutation_method=mutation_method
        self.mutation_probability=mutation_probability
        self.mutation_multi_limit=mutation_multi_limit
        self.recombination_method=recombination_method
        self.best_individual = max(self.population.individuals)
        self.best_score = self.best_individual.fitness

    def run_generation(self: Self) -> None:
        #Step 1: Selection
        parent_amount = max(1, len(self.population.individuals) * 2)
        # print(f"PARENT\n{self.population}")
        best = self.selection_method.select(
            population=self.population,
            amount=parent_amount,
        )
        # print(f"\n\nBEST\n{best}")

        #Step 2: Crossbreeding
        new_population=self.cross_method.cross(best, len(self.population.individuals))
        # print(f"\n\nCROSSBRED\n{new_population}")

        #Step 3: Mutations
        for individual in new_population.individuals:
            # print(f"BEFORE: {individual}")
            if self.mutation_method == MutationType.SINGLEGENE:
                individual.mutate_single(p=self.mutation_probability)
            elif self.mutation_method == MutationType.MULTI_LIMITED:
                if self.mutation_multi_limit is None:
                    raise ValueError("mutation_multi_limit is required for limited mutation")
                individual.mutate_multi_lim(p=self.mutation_probability, amount=self.mutation_multi_limit)
            elif self.mutation_method == MutationType.MULTI_COMPLETE:
                individual.mutate_complete(p=self.mutation_probability)
            elif self.mutation_method == MutationType.MULTI_UNIFORM:
                individual.mutate_multi_uniform(p=self.mutation_probability)
            # print(f"AFTER:  {individual}")
        # print(f"\n\nMUTATED\n{new_population}")
        #Step 4: Recombine populations
        if self.recombination_method == RecombinationType.ADDITIVE:
            combined = self.population.individuals + new_population.individuals
            self.population = Population(sorted(combined, reverse=True)[:len(self.population.individuals)])
        else:
            self.population = new_population
        # print(f"RECOMBINED {self.population}")
        self.best_individual = max(self.population.individuals)
        self.best_score = self.best_individual.fitness

    def run(self: Self, max_generations: int = 10000, target_score: float = 0.0, plot: bool = True) -> Tuple[IndividualT, float]:
        if plot:
            plt.ion()
            graph = plt.imshow(self.best_individual.render())
            plt.draw()
            plt.pause(1)
        for i in range(1, max_generations + 1):
            if self.best_score >= target_score:
                break
            self.run_generation()
            print(f"Generation {i} with MSE {-self.best_score:.4f}")
            if plot:
                graph.set_data(self.best_individual.render()) # type: ignore
                plt.draw()
                plt.pause(0.01)
        return (self.best_individual, self.best_score)
