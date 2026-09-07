"Population classes to run genetic algos"
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
import random
from typing import Any, Generic, List, Self, Sequence, Tuple, TypeVar

from numpy import ndarray

class MutationType(Enum):
    SINGLEGENE = "Single Gene"
    MULTI_LIMITED = "Multigene limited"
    MULTI_UNIFORM = "Multigene random uniform"
    MULTI_COMPLETE = "Multigene complete"

IndividualT = TypeVar("IndividualT", bound="Individual")
TargetT = TypeVar("TargetT", bound="Target")

class Target(ABC, Generic[IndividualT]):
    @abstractmethod
    def total_score(self: Self, population: Sequence[IndividualT]) -> float:
        "Returns the global score for this generation"


class Population(Generic[IndividualT]):
    def __init__(self: Self, individuals: List[IndividualT]):
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


class Individual(ABC, Generic[IndividualT, TargetT]):
    fitness: float
    genome_length: int

    @abstractmethod
    def swap_genes(self: Self, other: IndividualT, locuses: List[int]) -> Tuple[IndividualT, IndividualT]:
        "Swaps the gene at locuses provided between self and other and returns new children with those genes"

    @abstractmethod
    def mutate_genes(self: Self, positions: List[int]) -> None:
        "Mutates the gene at given positions"

    def mutate_single(self: Self, p: float):
        "Selects a random gene and mutatates it according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if random.random() <= p:
            self.mutate_genes([random.randrange(0, self.genome_length)])

    def mutate_multi_lim(self: Self, amount: int, p: float):
        "Selects up to amount random genes and mutatates them according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if amount < 1 or amount > self.genome_length:
            raise AttributeError("Amount of genes out of range")
        indexes = [random.randrange(0, self.genome_length) for _ in range(random.randrange(1, amount))]
        genes = []
        for i in indexes:
            if random.random() <= p:
                genes.append(i)
        self.mutate_genes(genes)

    def mutate_multi_uniform(self: Self, p: float):
        "Every gene can be mutated individually according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        genes = []
        for i in range(self.genome_length):
            if random.random() <= p:
                genes.append(i)
        self.mutate_genes(genes)

    def mutate_complete(self: Self, p: float):
        "The entire genome will mutate according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if random.random() <= p:
            self.mutate_genes(list(range(self.genome_length)))

IndividualFactoryT = TypeVar("IndividualFactoryT", bound="IndividualFactory")

class IndividualFactory(ABC, Generic[IndividualT]):
    @abstractmethod
    def create(self: Self) -> IndividualT:
        pass