from __future__ import annotations

from abc import ABC, abstractmethod
import random
from statistics import mean
from typing import Dict, Generic, List, Self, Sequence, Tuple, TypeVar, override

from PIL import Image as pimg, ImageDraw
import numpy as np
from numpy import ndarray

from gen.population import Individual, IndividualFactory, Target

Color = Tuple[int, int, int, int]
Point = Tuple[float, float]
Shapelike = TypeVar("Shapelike", bound="Shape")


class Shape(ABC):
    @abstractmethod
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        "Converts this shape into a SVG polygon"

    @abstractmethod
    def draw_on(self: Self, canvas: pimg.Image, width: int, height: int) -> None:
        "Alpha-composites this shape onto a Pillow canvas"

    @classmethod
    @abstractmethod
    def from_scratch(
        cls: type[Shapelike],
        target_image: ndarray | None = None,
    ) -> Triangle:
        pass

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
    def from_scratch(
        cls: type[Triangle],
        target_image: ndarray | None = None,
    ) -> Triangle:
        points = [cls.random_point() for _ in range(3)]
        if target_image is None or random.random() < 0.15:
            return cls(points, cls.random_color())
        return cls(points, cls.average_target_color(points, target_image))

    @staticmethod
    def average_target_color(points: List[Point], target_image: ndarray) -> Color:
        height, width = target_image.shape[:2]
        pixel_points = [
            (
                min(width - 1, int(x * width)),
                min(height - 1, int(y * height)),
            )
            for x, y in points
        ]
        mask = pimg.new("1", (width, height), 0)
        ImageDraw.Draw(mask).polygon(pixel_points, fill=1)
        covered_pixels = target_image[np.asarray(mask, dtype=bool)]
        if covered_pixels.size == 0:
            return Triangle.random_color()
        average_rgb = np.rint(covered_pixels[:, :3].mean(axis=0)).astype(int)
        return (
            int(average_rgb[0]),
            int(average_rgb[1]),
            int(average_rgb[2]),
            255,
        )

    def clone(self: Self) -> Triangle:
        return Triangle(list(self.points), self.color)

    def swap_genes(self: Self, other: Triangle, locuses: List[int]) -> None:
        mypoints, mycolor = self.listify()
        otherpoints, othercolor = other.listify()
        for locus in locuses:
            if locus < 6:
                point_index, coordinate = divmod(locus, 2)
                temp = mypoints[point_index][coordinate]
                mypoints[point_index][coordinate] = otherpoints[point_index][coordinate]
                otherpoints[point_index][coordinate] = temp
            else:
                color_index = locus - 6
                temp = mycolor[color_index]
                mycolor[color_index] = othercolor[color_index]
                othercolor[color_index] = temp
        self.apply_lists(mypoints, mycolor)
        other.apply_lists(otherpoints, othercolor)

    def listify(self: Self) -> Tuple[List[List[float]], List[int]]:
        points = [list(x) for x in self.points]
        color = list(self.color)
        return points, color

    def apply_lists(self: Self, points: List[List[float]], color: List[int]):
        self.points = [tuple(x) for x in points] # type: ignore[assignment]
        self.color = tuple(color) # type: ignore[assignment]
    
    def mutate_genes(self: Self, locuses: List[int]) -> None:
        points, color = self.listify()
        for locus in locuses:
            if locus < 6:
                point_index, coordinate = divmod(locus, 2)
                point = points[point_index]
                new_value = random.random()
                if new_value == point[coordinate]:
                    new_value = (new_value + 0.5) % 1
                points[point_index][coordinate] = new_value
            else:
                color_index = locus - 6
                new_value = random.randint(0, 255)
                if new_value == color[color_index]:
                    new_value = (new_value + 1) % 256
                color[color_index] = new_value
        self.apply_lists(points, color)

    def __hash__(self: Self) -> int:
        return hash((*self.points, self.color))

    @override
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        points = []
        for point in self.points:
            points.append(f"{int(point[0] * x)},{int(point[1] * y)}")
        return (
            f'<polygon points="{" ".join(points)}" '
            f'fill="#{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}" fill-opacity="{self.color[3]/255:.2f}"/>'
        )

    @override
    def draw_on(self: Self, canvas: pimg.Image, width: int, height: int) -> None:
        points = [
            (int(point[0] * width), int(point[1] * height))
            for point in self.points
        ]
        overlay = pimg.new("RGBA", (width, height), (0, 0, 0, 0))
        ImageDraw.Draw(overlay).polygon(points, fill=self.color)
        canvas.alpha_composite(overlay)

    def __str__(self: Self) -> str:
        return self.to_svg_polygon(1000,1000)
    def __repr__(self: Self) -> str:
        return str(self)

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
        target = self.image[:, :, :3].astype(np.float32)
        candidate = image[:, :, :3].astype(np.float32)
        return -float(np.mean(np.square(target - candidate)))
        

    def total_score(self: Self, population: Sequence[ImageIndividual]) -> float:
        return mean([i.fitness for i in population])


class ImageIndividual(Individual["ImageIndividual", TargetImage]):
    def __init__(self: Self, image: TargetImage, triangles: List[Triangle], evaluate: bool = True):
        self.target = image
        self.triangles = triangles
        self.genome_length = len(triangles) * Triangle.genome_length
        self.last_hash: int = hash(self)
        self.rendered: ndarray | None = None
        self.fitness = self.score() if evaluate else float("-inf")

    @classmethod
    def from_scratch(cls: type[ImageIndividual], shape: type[Shapelike], target: TargetImage, count: int) -> ImageIndividual:
        return cls(
            target,
            [shape.from_scratch(target.image) for _ in range(count)],
        )

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
        first_child = ImageIndividual(self.target, [triangle.clone() for triangle in self.triangles], evaluate=False)
        second_child = ImageIndividual(other.target, [triangle.clone() for triangle in other.triangles], evaluate=False)

        swaps: Dict[Tuple[Triangle, Triangle], List[int]] = {}

        for locus in locuses:
            t1, idx = first_child._locus(locus)
            t2, _ = second_child._locus(locus)
            if (t1, t2) in swaps:
                swaps[(t1, t2)].append(idx)
            else:
                swaps[(t1, t2)] = [idx]

        for (t1, t2), positions in swaps.items():
            t1.swap_genes(t2, positions)
        first_child.fitness = first_child.score()
        second_child.fitness = second_child.score()
        return first_child, second_child

    @override
    def mutate_genes(self: Self, locuses: List[int]) -> None:
        mutations: Dict[Triangle, List[int]] = {}
        for locus in locuses:
            triangle, gene_index = self._locus(locus)
            if triangle in mutations:
                mutations[triangle].append(gene_index)
            else:
                mutations[triangle] = [gene_index]
        for triangle, positions in mutations.items():
            triangle.mutate_genes(positions)
        self.fitness = self.score()

    def render(self: Self) -> ndarray:
        if hash(self) != self.last_hash or self.rendered is None:
            height, width = self.target.image.shape[:2]
            canvas = pimg.new("RGBA", (width, height), (255, 255, 255, 255))
            for shape in self.triangles:
                shape.draw_on(canvas, width, height)
            self.rendered = np.asarray(canvas.convert("RGB"))
            self.last_hash = hash(self)
        return self.rendered

    def score(self: Self) -> float:
        self.triangles = self.triangles
        return self.target.image_similarity(self.render())

    def __lt__(self: Self, other: ImageIndividual) -> bool:
        return self.fitness < other.fitness

    def __hash__(self: Self) -> int:
        return hash((*self.triangles,))

    def __str__(self: Self) -> str:
        return "INDIVIDUAL\n" + "\n".join(str(t) for t in self.triangles)

    def __repr__(self: Self) -> str:
        return str(self)

class ImageIndividualFactory(IndividualFactory, Generic[Shapelike]):
    def __init__(self: Self, target: TargetImage, shape: type[Shapelike], shape_count: int):
        self.shape_count=shape_count
        self.shape=shape
        self.target=target

    def create(self: Self) -> ImageIndividual:
        return ImageIndividual.from_scratch(target=self.target, shape=self.shape, count=self.shape_count)
