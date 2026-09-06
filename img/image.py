import random
from typing import List, Self, Tuple, override

from numpy import ndarray

from gen.population import Individual

class Triangle(Individual):
    def __init__(self: Self, points: List[Tuple[float, float]], color: Tuple[int, int, int, int]):
        # Points are tuples of floats (x, y) in range [0, 1]
        # When rasterizing, will stretch to img dimensions
        self.points: List[Tuple[float, float]] = sorted(points)
        # Colors are tuples of (r, g, b, a) in range [0, 255]
        # The most basic of colorspaces
        self.color: Tuple[int, int, int, int] = color
        self.genome_length = 10 #Total of 10 locus, hardcoded


    @staticmethod
    def random_point():
        return (random.random(), random.random())

    @staticmethod
    def random_color():
        return (
            random.randrange(0, 255),
            random.randrange(0, 255),
            random.randrange(0, 255),
            random.randrange(0, 255)
        )

    @classmethod
    @override
    def from_scratch(cls: type[Triangle]) -> Triangle:
        points = []
        for _ in range(3):
            points.append(cls.random_point())
        return cls(points, cls.random_color())

    def swap_gene(self: Self, other: Triangle, position: int) -> None:
        # Not the best, but it should do the trick
        if position < 6:
            temp = self.points[int(position / 2)]
            if not position % 2:
                self.points[int(position/2)] = (other.points[int(position/2)][0], temp[1])
                other.points[int(position/2)] = (temp[0], other.points[int(position/2)][1])
            else:
                self.points[int(position/2)] = (temp[0], other.points[int(position/2)][1])
                other.points[int(position/2)] = (other.points[int(position/2)][0], temp[1])
        else:
            temp = self.color
            position = (position-6) % 4
            if position == 0:
                self.color = (other.color[0], temp[1], temp[2], temp[3])
                other.color = (temp[0], other.color[1], other.color[2], other.color[3])
            elif position == 1:
                self.color = (temp[0], other.color[1], temp[2], temp[3])
                other.color = (other.color[0], temp[1], other.color[2], other.color[3])
            elif position == 2:
                self.color = (temp[0], temp[1], other.color[2], temp[3])
                other.color = (other.color[0], other.color[1], temp[2], other.color[3])
            elif position == 3:
                self.color = (temp[0], temp[1], temp[2], other.color[3])
                other.color = (other.color[0], other.color[1], other.color[2], temp[3])

    def mutate_gene(self: Self, position: int) -> None:
        if position < 6:
            if not position % 2:
                self.points[int(position/2)] = (random.random(), self.points[int(position/2)][1])
            else:
                self.points[int(position/2)] = (self.points[int(position/2)][0], random.random())
        else:
            position = (position-6) % 4
            newvalue = random.randrange(0,255)
            if position == 0:
                self.color = (newvalue, self.color[1], self.color[2], self.color[3])
            elif position == 1:
                self.color = (self.color[0], newvalue, self.color[2], self.color[3])
            elif position == 2:
                self.color = (self.color[0], self.color[1], newvalue, self.color[3])
            elif position == 3:
                self.color = (self.color[0], self.color[1], self.color[2], newvalue)

