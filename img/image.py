from __future__ import annotations

from abc import ABC, abstractmethod
import io
import random
from math import isfinite
from typing import List, Self, Sequence, Tuple, override

from PIL import Image as pimg
import numpy as np
from numpy import ndarray
from cairosvg import svg2png

from gen.population import Individual, Target

Color = Tuple[int, int, int, int]
Point = Tuple[float, float]


class Shape(ABC):
    @abstractmethod
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        "Converts this shape into a SVG polygon"


class Triangle(Shape):
    genome_length = 10

    def __init__(self: Self, points: List[Point], color: Color):
        self.points: List[Point] = list(points)
        self.color: Color = color

    @staticmethod
    def random_point() -> Point:
        return (random.random(), random.random())

    @staticmethod
    def random_color() -> Color:
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

    @classmethod
    def from_scratch(cls: type[Triangle]) -> Triangle:
        return cls([cls.random_point() for _ in range(3)], cls.random_color())

    def clone(self: Self) -> Triangle:
        return Triangle(self.points, self.color)

    def swap_gene(self: Self, other: Triangle, position: int) -> None:
        if not 0 <= position < self.genome_length:
            raise IndexError("Gene position out of range")
        if position < 6:
            point_index, coordinate = divmod(position, 2)
            self_point = self.points[point_index]
            other_point = other.points[point_index]
            if coordinate == 0:
                self.points[point_index] = (other_point[0], self_point[1])
                other.points[point_index] = (self_point[0], other_point[1])
            else:
                self.points[point_index] = (self_point[0], other_point[1])
                other.points[point_index] = (other_point[0], self_point[1])
            return

        color_index = position - 6
        self_color = list(self.color)
        other_color = list(other.color)
        self_color[color_index], other_color[color_index] = (
            other_color[color_index],
            self_color[color_index],
        )
        self.color = tuple(self_color)  # type: ignore[assignment]
        other.color = tuple(other_color)  # type: ignore[assignment]

    def mutate_gene(self: Self, position: int) -> None:
        if not 0 <= position < self.genome_length:
            raise IndexError("Gene position out of range")
        if position < 6:
            point_index, coordinate = divmod(position, 2)
            point = self.points[point_index]
            new_value = random.random()
            if new_value == point[coordinate]:
                new_value = (new_value + 0.5) % 1
            if coordinate == 0:
                self.points[point_index] = (new_value, point[1])
            else:
                self.points[point_index] = (point[0], new_value)
            return

        color_index = position - 6
        color = list(self.color)
        new_value = random.randint(0, 255)
        if new_value == color[color_index]:
            new_value = (new_value + 1) % 256
        color[color_index] = new_value
        self.color = tuple(color)  # type: ignore[assignment]

    @override
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        points = []
        for point in self.points:
            points.append(f"{int(point[0] * x)},{int(point[1] * y)}")
        return (
            f'<polygon points="{" ".join(points)}" '
            f'fill="#{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}" '
            f'fill-opacity="{self.color[3] / 255:.2f}"/>'
        )


class Image:
    def __init__(self: Self, image: ndarray):
        self.image = image

    def shapes_to_image(
        self: Self,
        shapes: Sequence[Shape],
        with_background: bool = True,
        background_color: Color = (255, 255, 255, 255),
    ) -> ndarray:
        height, width = self.image.shape[:2]
        svg = (
            f'<svg height="{height}" width="{width}" viewBox="0 0 {width} {height}" '
            'xmlns="http://www.w3.org/2000/svg">'
        )
        if with_background:
            svg += (
                f'<rect width="{width}" height="{height}" '
                f'fill="#{background_color[0]:02x}{background_color[1]:02x}{background_color[2]:02x}" '
                f'fill-opacity="{background_color[3] / 255:.2f}"/>'
            )
        for shape in shapes:
            svg += shape.to_svg_polygon(width, height)
        svg += "</svg>"
        svg_img = svg2png(svg, output_width=width, output_height=height)
        if svg_img:
            return np.array(pimg.open(io.BytesIO(svg_img)).convert("RGBA"))
        raise ValueError("Cannot generate image")

    def image_similarity(self: Self, image: ndarray) -> float:
        global_score = 0
        for y in range(self.image.shape[0]):
            for x in range(self.image.shape[1]):
                if image[y, x, 3] > 0:
                    pixel_score = 1.0
                    for channel in range(self.image.shape[2]):
                        pixel_score -= abs(
                            self.image[y, x, channel] - image[y, x, channel]
                        ) / (255 * 3)
                    global_score += pixel_score
        return global_score / (self.image.shape[0] * self.image.shape[1])


class ImageProblem(Target["ImageIndividual"]):
    def __init__(
        self: Self,
        target_image: Image,
        triangle_count: int,
        background_color: Color = (255, 255, 255, 255),
    ):
        if triangle_count < 1:
            raise ValueError("triangle_count must be at least 1")
        self.target_image = target_image
        self.triangle_count = triangle_count
        self.background_color = background_color

    @override
    def total_score(self: Self, population: Sequence[ImageIndividual]) -> float:
        if not population:
            raise ValueError("Population cannot be empty")
        return max(individual.fitness for individual in population)

    def shapes_to_image(self: Self, population: Sequence[ImageIndividual]) -> ndarray:
        if not population:
            raise ValueError("Population cannot be empty")
        return max(population, key=lambda individual: individual.fitness).render()


class ImageIndividual(Individual["ImageIndividual", ImageProblem]):
    def __init__(self: Self, problem: ImageProblem, triangles: List[Triangle]):
        if len(triangles) != problem.triangle_count:
            raise ValueError("The individual must contain exactly triangle_count triangles")
        self.problem = problem
        self.triangles = triangles
        self.genome_length = problem.triangle_count * Triangle.genome_length
        self.fitness = self.score()

    @classmethod
    @override
    def from_scratch(cls: type[ImageIndividual], target: ImageProblem) -> ImageIndividual:
        return cls(target, [Triangle.from_scratch() for _ in range(target.triangle_count)])

    def _locus(self: Self, position: int) -> Tuple[Triangle, int]:
        if not 0 <= position < self.genome_length:
            raise IndexError("Gene position out of range")
        triangle_index, gene_index = divmod(position, Triangle.genome_length)
        return self.triangles[triangle_index], gene_index

    @override
    def swap_genes(
        self: Self,
        other: ImageIndividual,
        locuses: List[int],
    ) -> Tuple[ImageIndividual, ImageIndividual]:
        if self.genome_length != other.genome_length:
            raise ValueError("Individuals must have the same genome length")
        first_child = ImageIndividual(self.problem, [triangle.clone() for triangle in self.triangles])
        second_child = ImageIndividual(other.problem, [triangle.clone() for triangle in other.triangles])
        for position in locuses:
            first_triangle, gene_index = first_child._locus(position)
            second_triangle, other_gene_index = second_child._locus(position)
            assert gene_index == other_gene_index
            first_triangle.swap_gene(second_triangle, gene_index)
        first_child.fitness = first_child.score()
        second_child.fitness = second_child.score()
        return first_child, second_child

    @override
    def mutate_gene(self: Self, position: int) -> None:
        triangle, gene_index = self._locus(position)
        triangle.mutate_gene(gene_index)
        self.fitness = self.score()

    def render(self: Self) -> ndarray:
        return self.problem.target_image.shapes_to_image(
            self.triangles,
            background_color=self.problem.background_color,
        )

    def score(self: Self) -> float:
        fitness = self.problem.target_image.image_similarity(self.render())
        if not isfinite(fitness):
            raise ValueError("Fitness must be finite")
        return fitness
