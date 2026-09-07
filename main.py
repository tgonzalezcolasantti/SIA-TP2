from PIL import Image as pimg
import numpy as np

from gen.cross import UniformCross
from gen.genetic import GeneticAlgorithm, RecombinationType
from gen.population import MutationType
from gen.selection import RouletteSelection
from img.image import Image, Triangle

def main():
    target = np.array(pimg.open("./flag_argentina.png").convert("RGB"))
    algo = GeneticAlgorithm(
        target=Image(target),
        initial_size=20,
        individual=Triangle,
        selection_method=RouletteSelection(),
        cross_method=UniformCross(),
        mutation_method=MutationType.MULTI_UNIFORM,
        mutation_probability=0.1,
        recombination_method=RecombinationType.EXCLUSIVE
    )
    ans, _ = algo.run(target_score=0.9)
    for triangle in ans:
        print(triangle.to_svg_polygon(target.shape[0], target.shape[1]))

if __name__ == "__main__":
    main()
