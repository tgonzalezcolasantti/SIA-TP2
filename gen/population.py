"Population classes to run genetic algos"
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
import random
from typing import Generic, List, Self, Tuple, TypeVar

class CrossType(Enum):
    ONEPOINT = "One Point"
    TWOPOINT = "Two Point"
    RING = "Ring"
    UNIFORM = "Uniform"

class MutationType(Enum):
    SINGLEGENE = "Single Gene"
    MULTI_LIMITED = "Multigene limited"
    MULTI_UNIFORM = "Multigene random uniform"
    MULTI_COMPLETE = "Multigene complete"

IndividualT = TypeVar("IndividualT", bound="Individual")

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


class Individual(ABC, Generic[IndividualT]):
    fitness: float
    genome_length: int

    @abstractmethod
    @classmethod
    def from_scratch(cls: type[Individual]) -> Individual:
        "Creates a new individual with random genes"

    @abstractmethod
    def swap_gene(self: Self, other: IndividualT, position: int) -> None:
        "Swaps the gene at locus position between self and other"

    @abstractmethod
    def mutate_gene(self: Self, position: int) -> None:
        "Mutates the gene at the given position"

    def cross_1p(self: Self, other: IndividualT, position: int):
        "Swaps genes from position until end of genome"
        #No need to raise index errors, indexing the array wrong will do it for us
        #Does that mean we can have negative indexes? Yes, it does. Doesn't matter though
        self.cross_2p(other, position, self.genome_length)

    def cross_2p(self: Self, other: IndividualT, p1: int, p2):
        "Swaps genes between p1 and p2"
        if p1 >= p2:
            raise AttributeError("P1 must be smaller than P2")
        for i in range(p1, p2):
            self.swap_gene(other, i % self.genome_length)

    def cross_ring(self: Self, other: IndividualT, position: int, length):
        "Swaps length genes starting at position and wraps around the end"
        if length > ceil(self.genome_length/2):
            raise AttributeError("Cannot swap more than half the genome")
        self.cross_2p(other, position, position + length)

    def cross_uniform(self: Self, other: IndividualT, p: float):
        "Swaps each gene according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        for i in range(self.genome_length):
            if random.random() >= p:
                self.swap_gene(other, i)

    @abstractmethod
    def mutate_single(self: Self, p: float):
        "Selects a random gene and mutatates it according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if random.random() >= p:
            self.mutate_gene(random.randrange(0, self.genome_length))

    @abstractmethod
    def mutate_multi_lim(self: Self, amount: int, p: float):
        "Selects up to amount random genes and mutatates them according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if amount < 1 or amount > self.genome_length:
            raise AttributeError("Amount of genes out of range")
        indexes = [random.randrange(0, self.genome_length) for _ in range(random.randrange(1, amount))]
        for i in indexes:
            if random.random() >= p:
                self.mutate_gene(i)

    @abstractmethod
    def mutate_multi_uniform(self: Self, p: float):
        "Every gene can be mutated individually according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        for i in range(self.genome_length):
            if random.random() >= p:
                self.mutate_gene(i)

    @abstractmethod
    def mutate_complete(self: Self, p: float):
        "The entire genome will mutate according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if random.random() >= p:
            for i in range(self.genome_length):
                self.mutate_gene(i)
