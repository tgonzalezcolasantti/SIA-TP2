from abc import ABC, abstractmethod
from typing import Any, Generic, List, Self, Tuple, TypeVar

class Population():
    def __init__(self: Self, individuals: List[Individual]):
        self.individuals = individuals
        self.total_fitness = sum(p.fitness for p in self.individuals)

    def relative_fitness(self: Self, individual: Individual)-> float:
        return individual.fitness / self.total_fitness

    def by_fitness(self: Self):
        return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

    def accumulated_relative_fitness(self: Self) -> List[Tuple[float, Individual]]:
        accumulator = 0
        result = []
        for p in self.individuals:
            accumulator += self.relative_fitness(p)
            result.append((accumulator, p))
        return result

GenotypeT = TypeVar("GenotypeT", bound="Any")

class Individual(ABC, Generic[GenotypeT]):
    genotype: GenotypeT
    fitness: float
    @abstractmethod
    @classmethod
    def from_scratch(cls: type[Individual]) -> Individual:
        "Creates a new individual with random genes"


    @abstractmethod
    def cross(self: Self):
        pass