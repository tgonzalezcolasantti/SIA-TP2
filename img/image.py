from __future__ import annotations

from abc import ABC, abstractmethod
import io
import random
from statistics import mean
from typing import Generic, List, Self, Sequence, Tuple, TypeVar, override

from PIL import Image as pimg
import numpy as np
from numpy import ndarray
from cairosvg import svg2png

from gen.population import Individual, IndividualFactory, Target

Color = Tuple[int, int, int]
Point = Tuple[float, float]
Shapelike = TypeVar("Shapelike", bound="Shape")


class Shape(ABC):
    @abstractmethod
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        "Converts this shape into a SVG polygon"

    @classmethod
    @abstractmethod
    def from_scratch(cls: type[Shapelike]) -> Triangle:
        pass

class Triangle(Shape):
    genome_length = 9

    def __init__(self: Self, points: List[Point], color: Color):
        self.points: List[Point] = list(points)
        self.color: Color = color
        self.rendered: ndarray | None = None
        self.last_hash: int = hash(self)

    @staticmethod
    def random_point() -> Point:
        return (random.random(), random.random())

    @staticmethod
    def random_color() -> Color:
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            #random.randint(0, 255),
        )

    @classmethod
    def from_scratch(cls: type[Triangle]) -> Triangle:
        return cls([cls.random_point() for _ in range(3)], cls.random_color())

    def clone(self: Self) -> Triangle:
        return Triangle(list(self.points), self.color)

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
    
    def __hash__(self: Self) -> int:
        return hash((*self.points, self.color))

    @override
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        points = []
        for point in self.points:
            points.append(f"{int(point[0] * x)},{int(point[1] * y)}")
        return (
            f'<polygon points="{" ".join(points)}" '
            f'fill="#{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}"/>'
        )

class TargetImage(Target["ImageIndividual"]):
    def __init__(self: Self, image: ndarray):
        self.image = image

    # def image_similarity(self: Self, image: ndarray) -> float:
    #     global_score = 0
    #     for y in range(self.image.shape[0]):
    #         for x in range(self.image.shape[1]):
    #             # if image[y, x, 3] > 0:
    #             pixel_score = 1.0
    #             for channel in range(self.image.shape[2]):
    #                 pixel_score -= abs(
    #                     self.image[y, x, channel] - image[y, x, channel]
    #                 ) / (255 * 3)
    #             global_score += pixel_score ** 4
    #     return global_score / (self.image.shape[0] * self.image.shape[1])
    def image_similarity(self: Self, image: ndarray) -> float:
        return np.sum((1-np.abs((np.subtract(self.image.ravel(),image.ravel(),dtype=np.int16))/255))) / (np.prod(self.image.shape))
        

    def total_score(self: Self, population: Sequence[ImageIndividual]) -> float:
        return mean([i.fitness for i in population])


class ImageIndividual(Individual["ImageIndividual", TargetImage]):
    def __init__(self: Self, image: TargetImage, triangles: List[Triangle]):
        self.target = image
        self.triangles = triangles
        self.genome_length = len(triangles) * Triangle.genome_length
        self.last_hash: int = hash(self)
        self.rendered: ndarray | None = None
        self.fitness = self.score()

    @classmethod
    def from_scratch(cls: type[ImageIndividual], shape: type[Shapelike], target: TargetImage, count: int) -> ImageIndividual:
        return cls(target, [shape.from_scratch() for _ in range(count)])

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
        first_child = ImageIndividual(self.target, [triangle.clone() for triangle in self.triangles])
        second_child = ImageIndividual(other.target, [triangle.clone() for triangle in other.triangles])
        for position in locuses:
            first_triangle, gene_index = first_child._locus(position)
            second_triangle, other_gene_index = second_child._locus(position)
            assert gene_index == other_gene_index
            first_triangle.swap_gene(second_triangle, gene_index)
        first_child.fitness = first_child.score()
        second_child.fitness = second_child.score()
        return first_child, second_child

    @override
    def mutate_genes(self: Self, positions: List[int]) -> None:
        for position in positions:
            triangle, gene_index = self._locus(position)
            triangle.mutate_gene(gene_index)
        self.fitness = self.score()

    def render(self: Self) -> ndarray:
        if hash(self) != self.last_hash or self.rendered is None:
            height, width = self.target.image.shape[:2]
            svg = (
                f'<svg height="{height}" width="{width}" viewBox="0 0 {width} {height}" '
                'xmlns="http://www.w3.org/2000/svg">'
            )
            svg += (
                f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
            )
            for shape in self.triangles:
                svg += shape.to_svg_polygon(width, height)
            svg += "</svg>"
            svg_img = svg2png(svg, output_width=width, output_height=height)
            if svg_img:
                self.rendered = np.array(pimg.open(io.BytesIO(svg_img)).convert("RGB"))
                self.last_hash = hash(self)
            else:
                raise ValueError("Cannot generate image")
        return self.rendered

    def score(self: Self) -> float:
        return self.target.image_similarity(self.render())

    def __lt__(self: Self, other: ImageIndividual) -> bool:
        return self.fitness < other.fitness

    def __hash__(self: Self) -> int:
        return hash((*self.triangles,))

class ImageIndividualFactory(IndividualFactory, Generic[Shapelike]):
    def __init__(self: Self, target: TargetImage, shape: type[Shapelike], shape_count: int):
        self.shape_count=shape_count
        self.shape=shape
        self.target=target

    def create(self: Self) -> ImageIndividual:
        return ImageIndividual.from_scratch(target=self.target, shape=self.shape, count=self.shape_count)
