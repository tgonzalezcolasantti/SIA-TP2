from abc import ABC, abstractmethod
from itertools import permutations
from math import ceil
import random
from typing import Self

from gen.population import Population


class CrossMethod(ABC):
    @abstractmethod
    def cross(self: Self, population: Population, children: int) -> Population:
        "Creates a new population by crossbreeding current population"

class OnePointCross(CrossMethod):
    "Swaps genes from a random position until end of genome"
    def cross(self: Self, population: Population, children: int):
        new_individuals = []
        length = len(population.individuals)
        for _ in range(0, children, 2):
            p1 = population.individuals[random.randrange(0, length)]
            p2 = population.individuals[random.randrange(0, length)]
            start_locus = random.randrange(0, p1.genome_length)
            locuses = list(range(start_locus, p1.genome_length))
            new_individuals.extend(p1.swap_genes(p2, locuses))
        return Population(new_individuals)

class TwoPointCross(CrossMethod):
    "Swaps genes between p1 and p2"
    def cross(self: Self, population: Population, children: int) -> Population:
        new_individuals = []
        length = len(population.individuals)
        for _ in range(0, children, 2):
            p1 = population.individuals[random.randrange(0, length)]
            p2 = population.individuals[random.randrange(0, length)]
            start_locus = random.randrange(0, p1.genome_length)
            end_locus = random.randrange(start_locus, p1.genome_length)
            locuses = list(range(start_locus, end_locus))
            new_individuals.extend(p1.swap_genes(p2, locuses))
        return Population(new_individuals)

class RingCross(CrossMethod):
    "Swaps length genes starting at position and wraps around the end"
    def cross(self: Self, population: Population, children: int) -> Population:
        new_individuals = []
        length = len(population.individuals)
        for _ in range(0, children, 2):
            p1 = population.individuals[random.randrange(0, length)]
            p2 = population.individuals[random.randrange(0, length)]
            start_locus = random.randrange(0, p1.genome_length)
            length = random.randrange(0, ceil(p1.genome_length/2))
            locuses = [i % p1.genome_length for i in range(start_locus, start_locus + length)]
            new_individuals.extend(p1.swap_genes(p2, locuses))
        return Population(new_individuals)

class UniformCross(CrossMethod):
    "Swaps each gene according to probability p"
    def cross(self: Self, population: Population, children: int) -> Population:
        new_individuals = []
        length = len(population.individuals)
        for _ in range(0, children, 2):
            probability = random.random()
            p1 = population.individuals[random.randrange(0, length)]
            p2 = population.individuals[random.randrange(0, length)]
            locuses = [i for i in range(p1.genome_length) if random.random() >= probability]
            new_individuals.extend(p1.swap_genes(p2, locuses))
        return Population(new_individuals)
