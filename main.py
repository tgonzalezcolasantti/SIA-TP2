import random

from PIL import Image as pimg
import numpy as np

from gen.cross import UniformCross
from gen.genetic import GeneticAlgorithm, RecombinationType
from gen.population import MutationType
from gen.selection import BoltzmannSelection, RouletteSelection
from img.image import ImageIndividualFactory, TargetImage, TargetImage, Triangle

def main():
    random.seed(1234)
    target = np.array(pimg.open("./flag_argentina.png").convert("RGB"))
    problem = TargetImage(target)
    algo = GeneticAlgorithm(
        target=problem,
        initial_size=100,
        individual_factory=ImageIndividualFactory(target=problem, shape=Triangle, shape_count=20),
        selection_method=BoltzmannSelection(initial_temp=1, target_temp=0.01, rate=0.05),
        cross_method=UniformCross(),
        mutation_method=MutationType.MULTI_UNIFORM,
        mutation_probability=0.01,
        recombination_method=RecombinationType.ADDITIVE
    )
    ans, _ = algo.run(target_score=0.9)
    for triangle in ans.triangles:
        print(triangle.to_svg_polygon(target.shape[1], target.shape[0]))

if __name__ == "__main__":
    main()
