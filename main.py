import random

from PIL import Image as pimg
import numpy as np

from gen.cross import UniformCross
from gen.genetic import GeneticAlgorithm, RecombinationType
from gen.population import MutationType
from gen.selection import RouletteSelection
from img.image import ImageIndividualFactory, TargetImage, Triangle

def main():
    random.seed(1234)
    target = np.array(pimg.open("./flag_argentina.png").convert("RGB"))
    problem = TargetImage(target)
    algo = GeneticAlgorithm(
        target=problem,
        initial_size=20,
        individual_factory=ImageIndividualFactory(target=problem, shape=Triangle, shape_count=20),
        selection_method=RouletteSelection(),
        cross_method=UniformCross(),
        mutation_method=MutationType.SINGLEGENE,
        mutation_probability=0.2,
        recombination_method=RecombinationType.ADDITIVE
    )
    ans, score = algo.run(max_generations=20000, target_score=0.0)
    print(f"Final MSE: {-score:.6f}")
    pimg.fromarray(ans.render()).save("best_result.png")
    print("Best image saved to best_result.png")
    for triangle in ans.triangles:
        print(triangle.to_svg_polygon(target.shape[1], target.shape[0]))

if __name__ == "__main__":
    main()
