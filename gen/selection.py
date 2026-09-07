"Different selection methods to iterate across generations"

from abc import ABC, abstractmethod
from math import ceil, exp
from statistics import mean
import random
from typing import List, Optional, Self, Tuple, override

from gen.population import Individual, Population

class SelectionMethod(ABC):
    "Base class for genetic selection methods"
    @abstractmethod
    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        "Selects a given amount of individuals from a population"

class EliteSelection(SelectionMethod):
    "Selects the best individuals proportionally."
    @override
    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        result = []
        individuals = population.by_fitness()
        length = len(individuals)
        for idx, p in enumerate(individuals):
            times = ceil((amount - idx) / length)
            if times == 0:
                break
            result.extend([p for _ in range(times)])
        return result

class RandomSelection(SelectionMethod):
    "Selects random individuals"
    @override
    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        return random.choices(population.individuals, k=amount)

class SlotSelection(SelectionMethod, ABC):
    "Abstract class for slot-like selection methods"
    def slot_pick(
        self: Self,
        arf_population: List[Tuple[float, Individual]],
        slots: List[float]
    ) -> List[Individual]:
        """Picks individuals with scores that fall in slot transitions.
        aka the first individual with a score greater than the current slot position"""
        if not slots:
            return []
        if not arf_population:
            raise ValueError("Cannot select from an empty population")
        result = []
        individual_index = 0
        for slot in sorted(slots):
            while (
                individual_index < len(arf_population) - 1
                and arf_population[individual_index][0] < slot
            ):
                individual_index += 1
            result.append(arf_population[individual_index][1])
        return result

class RouletteSelection(SlotSelection, SelectionMethod):
    "Slot selection with random slots"
    @override
    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        return self.slot_pick(
            arf_population=population.accumulated_relative_fitness(),
            slots=self.generate_slots(amount)
        )

    @staticmethod
    def generate_slots(amount: int) -> List[float]:
        "Generates random slots for roulette-like selections"
        return sorted(random.random() for _ in range(amount))

class UniversalSelection(SlotSelection, SelectionMethod):
    "Slot selection with evenly-spaced slots with a random offset"
    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        if amount == 0:
            return []
        rand = random.random()
        return self.slot_pick(
            arf_population=population.accumulated_relative_fitness(),
            slots=[(rand + i) / amount for i in range(amount)]
        )

class PseudoFitnessSelection(RouletteSelection, SelectionMethod, ABC):
    "Performs roulette-like selections using a custom-defined pseudofitness score"
    @abstractmethod
    def pseudofitness(self: Self, idx: int, population: List[Individual]) -> float:
        "Calculates the pseudofitness value to use in roulette selection instead of fitness"

    @abstractmethod
    def prepare_next(self: Self) -> None:
        "Performs any necessary preparations for the next selection event, if necessary"

    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        by_fitness = population.by_fitness()
        scores = [self.pseudofitness(idx, by_fitness) for idx in range(len(by_fitness))]
        total = sum(scores)
        if scores and total <= 0:
            scores = [1.0] * len(scores)
            total = len(scores)
        accumulator = 0.0
        accumulated_scores = []
        for score, individual in zip(scores, by_fitness):
            accumulator += score / total
            accumulated_scores.append((accumulator, individual))
        if accumulated_scores:
            accumulated_scores[-1] = (1.0, accumulated_scores[-1][1])
        ans = self.slot_pick(
            arf_population=accumulated_scores,
            slots = self.generate_slots(amount)
        )
        self.prepare_next()
        return ans

class RankingSelection(PseudoFitnessSelection, SelectionMethod):
    "Roulette-lile selection using ranking position score as pseudofitness"
    def __init__(self: Self):
        self.length: Optional[int] = None

    @override
    def pseudofitness(self: Self, idx: int, population: List[Individual]) -> float:
        if not self.length:
            self.length = len(population)
        return (self.length - idx) / self.length

    @override
    def prepare_next(self: Self) -> None:
        self.length = None

class BoltzmannSelection(PseudoFitnessSelection, SelectionMethod):
    "Performs Boltzmann selection using Expected Value as pseudoscore"
    def __init__(self: Self, initial_temp: float, target_temp: float, rate: float):
        self.initial_temp = initial_temp
        self.target_temp = target_temp
        self.rate = rate
        self.current_temp = initial_temp
        self.iter = 0
        self.population_exp: Optional[List[float]] = None
        self.avg_exp: float = 0.0

    def individual_exp(self: Self, individual: Individual) -> float:
        "Calculates the individual exponent function based on fitness"
        return exp(individual.fitness / self.current_temp)

    @override
    def pseudofitness(self: Self, idx: int, population: List[Individual]) -> float:
        if not self.population_exp:
            maximum_fitness = max(individual.fitness for individual in population)
            self.population_exp = [
                exp((individual.fitness - maximum_fitness) / self.current_temp)
                for individual in population
            ]
            self.avg_exp = mean(self.population_exp)
        return self.population_exp[idx] / self.avg_exp

    @override
    def prepare_next(self: Self) -> None:
        self.population_exp = None
        self.iter += 1
        delta_temp = self.initial_temp - self.target_temp
        self.current_temp = self.target_temp + delta_temp * exp(-self.rate * self.iter)

class DeterministicTournamentSelection(SelectionMethod):
    """Performs deterministic tournaments between tournament_size participants
    and picks the best of each tournament."""
    def __init__(self: Self, tournament_size: int):
        self.tournament_size = tournament_size

    @override
    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        return [max(
            random.choices(population.individuals, k=self.tournament_size),
            key=lambda x: x.fitness
        ) for _ in range(amount)]

class ProbabilisticTournamentSelection(SelectionMethod):
    "Performs probabilistic duels, returning winners based on threshold level."
    def __init__(self: Self, threshold: float):
        if threshold > 1 or threshold < 0.5:
            raise AttributeError("Invalid threshold value, must be between 0.5 and 1")
        self.threshold = threshold

    @override
    def select(self: Self, population: Population, amount: int) -> List[Individual]:
        ans = []
        for _ in range(amount):
            rand = random.random()
            duel = random.choices(population.individuals, k=2)
            if rand < self.threshold:
                ans.append(max(duel, key=lambda x:x.fitness))
            else:
                ans.append(min(duel, key=lambda x:x.fitness))
        return ans
