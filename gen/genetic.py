from typing import Generic, Self

from gen.cross import CrossMethod
from gen.population import IndividualT, MutationType, TargetT, Population
from gen.selection import SelectionMethod

class GeneticAlgorithm(Generic[IndividualT, TargetT]):
    def __init__(
        self: Self,
        target: TargetT,
        initial_size: int,
        individual: type[IndividualT],
        selection_method: SelectionMethod,
        cross_method: CrossMethod,
        mutation_method: MutationType,
    ):
        self.population: Population = Population([individual.from_scratch(target) for _ in range(initial_size)])
        self.selection_method = selection_method
