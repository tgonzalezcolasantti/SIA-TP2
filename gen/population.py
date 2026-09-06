from abc import ABC, abstractmethod
from math import ceil
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
    genome_length: int
    @abstractmethod
    @classmethod
    def from_scratch(cls: type[Individual]) -> Individual:
        "Creates a new individual with random genes"


    @abstractmethod
    def cross_1p(self: Self, other: Individual, position: int):
        #No need to raise index errors, indexing the array wrong will do it for us
        #Does that mean we can have negative indexes? Yes, it does. Doesn't matter though
        pass
    @abstractmethod
    def cross_2p(self: Self, other: Individual, p1: int, p2):
        if p1 >= p2:
            raise AttributeError("P1 must be smaller than P2")

    @abstractmethod
    def cross_ring(self: Self, other: Individual, position: int, length):
        if length > ceil(self.genome_length/2):
            raise AttributeError("Cannot swap more than half the genome")

    @abstractmethod
    def cross_uniform(self: Self, other: Individual, p: float):
        pass

    