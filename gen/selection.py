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
            times = ceil(amount * idx / length)
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
        idx = 0
        result = []
        for fitness, individual in arf_population[idx:]:
            try:
                if fitness >= slots[0]:
                    slots.pop(0)
                    result.append(individual)
                else:
                    idx += 1
            except IndexError:
                break
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
        rand = random.random()
        return self.slot_pick(
            arf_population=population.accumulated_relative_fitness(),
            slots=[rand + i / amount for i in range(amount)]
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
        by_pseudo_fitness = sorted(
            [(self.pseudofitness(idx, by_fitness), p) for idx, p in enumerate(by_fitness)],
            key=lambda x: x[0],
            reverse=True
        )
        ans = self.slot_pick(
            arf_population=by_pseudo_fitness,
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
        return self.length - idx + 1 / self.length

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
            self.population_exp = [self.individual_exp(p) for p in population]
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
