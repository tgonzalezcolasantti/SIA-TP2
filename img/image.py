from abc import ABC, abstractmethod
import io
import random
from typing import List, Self, Tuple, override

from PIL import Image as pimg
import numpy as np
from numpy import ndarray
from cairosvg import svg2png

from gen.population import Individual

class Shape(Individual, ABC):
    @abstractmethod
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        "Converts this shape into a SVG polygon"

class Triangle(Shape, Individual):
    def __init__(self: Self, target: Image, points: List[Tuple[float, float]], color: Tuple[int, int, int, int]):
        # Points are tuples of floats (x, y) in range [0, 1]
        # When rasterizing, will stretch to img dimensions
        self.points: List[Tuple[float, float]] = sorted(points)
        # Colors are tuples of (r, g, b, a) in range [0, 255]
        # The most basic of colorspaces
        self.color: Tuple[int, int, int, int] = color
        self.genome_length = 10 #Total of 10 locus, hardcoded
        self.target = target
        self.fitness = self.score()

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
    def from_scratch(cls: type[Triangle], target: Image) -> Triangle:
        points = []
        for _ in range(3):
            points.append(cls.random_point())
        return cls(target, points, cls.random_color())

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

    @override
    def to_svg_polygon(self: Self, x: int, y: int) -> str:
        points = []
        for p in self.points:
            points.append(f'{int(p[0] * x)},{p[1] * y}')
        return f'<polygon points="{" ".join(points)}" fill="#{self.color[0]:x}{self.color[1]:x}{self.color[2]:x}" fill-opacity={self.color[3]/255:.2f}/>'

    def score(self: Self) -> float:
        newimage = self.target.shapes_to_image([self], with_background=False)
        return self.target.image_similarity(newimage)


class Image():
    def __init__(self: Self, image: ndarray):
        self.image = image

    def shapes_to_image(self: Self, shapes: List[Shape], with_background: bool = True) -> ndarray:
        x = self.image.shape[0]
        y = self.image.shape[1]
        svg = f'<svg height="{y}" width="{x}" viewBox="0 0 {x} {y}" xmlns="http://www.w3.org/2000/svg">'
        for s in shapes:
            svg += s.to_svg_polygon(x, y)
        svg += '</svg>'
        try:
            if with_background:
                svg_img = svg2png(svg, output_width=self.image.shape[0], output_height=self.image.shape[1], background_color="white")
            else:
                svg_img = svg2png(svg, output_width=self.image.shape[0], output_height=self.image.shape[1])
            if svg_img:
                raster = np.array(pimg.open(io.BytesIO(svg_img)).convert('RGBA'))
                return raster
            raise ValueError("Cannot generate image")
        except Exception as e:
            raise e

    def image_similarity(self: Self, image: ndarray) -> float:
        global_score = 0
        for x in range(self.image.shape[0]):
            for y in range(self.image.shape[1]):
                if image[x,y,3] > 0:
                    pixel_score = 1
                    for p in range(self.image.shape[2]):
                        pixel_score -= abs(self.image[x,y,p] - image[x,y,p]) / (255*3)
                    global_score += pixel_score
        return global_score / (self.image.shape[0] * self.image.shape[1])

    def shapes_score(self: Self, shapes: List[Shape]) -> float:
        newimage = self.shapes_to_image(shapes)
        return self.image_similarity(newimage)