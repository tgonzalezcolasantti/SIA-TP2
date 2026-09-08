import random

from PIL import Image as pimg
import numpy as np

from cli import (
    create_crossbreed,
    create_mutation,
    create_selection,
    create_survival,
    parse_args,
)
from gen.genetic import GeneticAlgorithm
from img.image import ImageIndividualFactory, TargetImage, Triangle


def main():
    args = parse_args()
    random.seed(args.seed)
    with pimg.open(args.image) as image:
        target = np.array(image.convert("RGB"))
    problem = TargetImage(target)
    algo = GeneticAlgorithm(
        target=problem,
        initial_size=args.population_size,
        individual_factory=ImageIndividualFactory(
            target=problem,
            shape=Triangle,
            shape_count=args.triangle_count,
        ),
        selection_method=create_selection(args),
        cross_method=create_crossbreed(args.crossbreed),
        mutation_method=create_mutation(args.mutation),
        mutation_probability=args.mutation_probability,
        recombination_method=create_survival(args.survival),
        mutation_multi_limit=args.mutation_limit,
    )
    ans, score = algo.run(
        max_generations=args.max_generations,
        target_score=-args.target_error,
        plot=not args.no_plot,
    )
    print(f"Final MSE: {-score:.6f}")
    if not args.no_plot:
        pimg.fromarray(ans.render()).save("best_result.png")
        print("Best image saved to best_result.png")
    for triangle in ans.triangles:
        print(triangle.to_svg_polygon(target.shape[1], target.shape[0]))


if __name__ == "__main__":
    main()
