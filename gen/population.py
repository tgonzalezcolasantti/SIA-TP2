"Population classes to run genetic algos"
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
import random
from typing import Generic, List, Self, Sequence, Tuple, TypeVar

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

    @property
    def total_fitness(self: Self) -> float:
        return sum(individual.fitness for individual in self.individuals)

    def _selection_weights(self: Self) -> List[float]:
        if not self.individuals:
            return []
        fitnesses = [float(individual.fitness) for individual in self.individuals]
        minimum = min(fitnesses)
        if minimum < 0:
            fitnesses = [fitness - minimum for fitness in fitnesses]
        total = sum(fitnesses)
        if total <= 0:
            return [1 / len(fitnesses)] * len(fitnesses)
        return [fitness / total for fitness in fitnesses]

    def relative_fitness(self: Self, individual: Individual)-> float:
        try:
            index = self.individuals.index(individual)
        except ValueError as error:
            raise ValueError("Individual does not belong to this population") from error
        return self._selection_weights()[index]

    def by_fitness(self: Self):
        return sorted(self.individuals, key=lambda x: x.fitness, reverse=True)

    def accumulated_relative_fitness(self: Self) -> List[Tuple[float, Individual]]:
        accumulator = 0
        result = []
        for individual, weight in zip(self.individuals, self._selection_weights()):
            accumulator += weight
            result.append((accumulator, individual))
        if result:
            result[-1] = (1.0, result[-1][1])
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

    def mutate_single(self: Self, p: float) -> bool:
        "Selects a random gene and mutatates it according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if random.random() <= p:
            self.mutate_genes([random.randrange(0, self.genome_length)])
            return True
        return False

    def mutate_multi_lim(self: Self, amount: int, p: float) -> bool:
        "Selects up to amount random genes and mutatates them according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if amount < 1 or amount > self.genome_length:
            raise AttributeError("Amount of genes out of range")
        indexes = random.sample(
            range(self.genome_length),
            k=random.randint(1, amount),
        )
        genes = [index for index in indexes if random.random() <= p]
        if genes:
            self.mutate_genes(genes)
            return True
        return False

    def mutate_multi_uniform(self: Self, p: float) -> bool:
        "Every gene can be mutated individually according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        genes = []
        for i in range(self.genome_length):
            if random.random() <= p:
                genes.append(i)
        if genes:
            self.mutate_genes(genes)
            return True
        return False

    def mutate_complete(self: Self, p: float) -> bool:
        "The entire genome will mutate according to probability p"
        if p < 0 or p > 1:
            raise AttributeError("Probability out of range")
        if random.random() <= p:
            self.mutate_genes(list(range(self.genome_length)))
            return True
        return False

IndividualFactoryT = TypeVar("IndividualFactoryT", bound="IndividualFactory")

class IndividualFactory(ABC, Generic[IndividualT]):
    @abstractmethod
    def create(self: Self) -> IndividualT:
        pass
