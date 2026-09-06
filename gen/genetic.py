from typing import Generic, Self, TypeVar

from gen.population import Individual, Population
from gen.selection import SelectionMethod

IndividualT = TypeVar("IndividualT", bound="Individual")

class GeneticAlgorithm(Generic[IndividualT]):
    def __init__(self: Self, initial_size: int, individual: type[IndividualT], selection_method: SelectionMethod):
        self.population: Population = Population([individual.from_scratch() for _ in range(initial_size)])
        self.selection_method = selection_method
