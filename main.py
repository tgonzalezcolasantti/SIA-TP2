from PIL import Image as pimg
import numpy as np

from gen.cross import UniformCross
from gen.genetic import GeneticAlgorithm, RecombinationType
from gen.population import MutationType
from gen.selection import RouletteSelection
from img.image import ImageIndividualFactory, TargetImage, ImageIndividual, TargetImage, Triangle

def main():
    target = np.array(pimg.open("./flag_argentina.png").convert("RGB"))
    problem = TargetImage(target)
    algo = GeneticAlgorithm(
        target=problem,
        initial_size=20,
        individual_factory=ImageIndividualFactory(target=problem, shape=Triangle, shape_count=20),
        selection_method=RouletteSelection(),
        cross_method=UniformCross(),
        mutation_method=MutationType.MULTI_UNIFORM,
        mutation_probability=0.1,
        recombination_method=RecombinationType.EXCLUSIVE
    )
    ans, _ = algo.run(target_score=0.9)
    for individual in ans:
        for triangle in individual.triangles:
            print(triangle.to_svg_polygon(target.shape[1], target.shape[0]))

if __name__ == "__main__":
    main()
