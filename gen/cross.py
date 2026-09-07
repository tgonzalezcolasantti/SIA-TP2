from abc import ABC, abstractmethod
from math import ceil
import random
from typing import List, Self, override

from gen.population import Individual, Population


class CrossMethod(ABC):
    @abstractmethod
    def get_locuses(self: Self, genome_length: int) -> List[int]:
        "Generates a list of locuses to swap between genes. This determines which method it is"

    def cross(self: Self, population: List[Individual], children: int) -> Population:
        "Crossbreeds current population to generate requested children and returns a new population with them"
        new_individuals = []
        length = len(population)
        for _ in range(0, children, 2):
            p1 = population[random.randrange(0, length)]
            p2 = population[random.randrange(0, length)]
            new_individuals.extend(p1.swap_genes(p2, self.get_locuses(p1.genome_length)))
        return Population(new_individuals)

class BadCross(CrossMethod):
    "Doesnt't really do anything"
    @override
    def get_locuses(self: Self, genome_length: int) -> List[int]:
        return []

class OnePointCross(CrossMethod):
    "Swaps genes from a random position until end of genome"
    @override
    def get_locuses(self: Self, genome_length: int) -> List[int]:
        start_locus = random.randrange(0, genome_length)
        return list(range(start_locus, genome_length))

class TwoPointCross(CrossMethod):
    "Swaps genes between p1 and p2"
    @override
    def get_locuses(self: Self, genome_length: int):
        start_locus = random.randrange(0, genome_length)
        end_locus = random.randrange(start_locus, genome_length)
        return list(range(start_locus, end_locus))

class RingCross(CrossMethod):
    "Swaps length genes starting at position and wraps around the end"
    @override
    def get_locuses(self: Self, genome_length: int) -> List[int]:
        start_locus = random.randrange(0, genome_length)
        length = random.randrange(0, ceil(genome_length/2))
        return [i % genome_length for i in range(start_locus, start_locus + length)]

class UniformCross(CrossMethod):
    "Swaps each gene according to probability p"
    @override
    def get_locuses(self: Self, genome_length: int) -> List[int]:
        probability = random.random()
        return [i for i in range(genome_length) if random.random() <= probability]
