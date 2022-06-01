from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from enum import Enum, auto
from typing import Callable
import cv2
import concurrent.futures
from functools import partial
from time import sleep, perf_counter as pc

from blur_detection.detection import detect_blur_fft, estimate_blur, fix_image_size
from file_handling import find_images, load_image


class Sharpness(Enum):
    SHARP = auto()
    SEMI = auto()
    BLURRY = auto()


@dataclass
class Algorithm:
    name: str
    call_func: Callable
    blurriness_threshold: int
    bl_params: dict = field(default_factory=dict)  # Bilateral filtering parameters
    hit_count: int = 0



@dataclass
class TestImage:
    path: Path
    expect_result: Sharpness
    actual_results: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)
    times: dict = field(default_factory=dict)

    def scores_str(self, algorithms: dict[str, Algorithm]):
        output_elements = []
        for algo, score in self.scores.items():
            result = self.actual_results[algo]
            hit_str = "HIT" if self.expect_result == result else "MISS"
            output_elements.append(f"{algo}={score:6.1f}={result.name:6}-{hit_str:4} ({self.times[algo]:.1f}s)")
        return " - ".join(output_elements)

    def result_str(self, algorithms: dict[str, Algorithm]):
        return f"{self.path.name:>20} {self.expect_result.name:6} --- {self.scores_str(algorithms)}"


def build_algos():
    output = {
        "FFT": Algorithm("FFT", detect_blur_fft, blurriness_threshold=20, bl_params={"pass_count":0}),
    }
    laplace_param_sets = [ # pass_count, diameter, colorSigma, spaceSigma
        (0,(0,0,0)),
        (1,(7,75,75)),
        (2,(7,75,75)),
        (5,(5,75,75)),
        (2,(7,150,150))
    ]
    for lp_params in laplace_param_sets:
        ident = f"Laplace({lp_params[0]}p-{lp_params[1][0]}-{lp_params[1][1]}-{lp_params[1][2]})"
        output[ident] = Algorithm(ident,
                                      partial(estimate_blur, just_score=True),
                                      blurriness_threshold=10,
                                      bl_params={
                                          "pass_count":lp_params[0],
                                          "filter_inputs": lp_params[1]
                                      }
                                  )
    return output

def test_an_image(test_im: TestImage, algorithms: dict[str, Algorithm]):
    image = load_image(test_im.path)
    image = fix_image_size(image, 3e6)
    for algo in algorithms.values():
        t0 = pc()

        i = 0
        while i < algo.bl_params["pass_count"]:
            image = cv2.bilateralFilter(image, *algo.bl_params["filter_inputs"])
            i += 1

        score = algo.call_func(image)
        elapsed = pc() - t0

        test_im.scores[algo.name] = score
        test_im.times[algo.name] = elapsed

        sharp: bool = score >= algorithms[algo.name].blurriness_threshold
        sharpness = Sharpness.SHARP if sharp else Sharpness.BLURRY
        test_im.actual_results[algo.name] = sharpness
    return test_im

def print_result(test_im: TestImage, algorithms: dict[str, Algorithm]):
    print(test_im.result_str(algorithms))

def main():
    test_sources = {
        Sharpness.SHARP: r"C:\Users\jinks\Desktop\Export\blurdet\sharp",
        Sharpness.BLURRY: r"C:\Users\jinks\Desktop\Export\blurdet\blurry"
    }

    test_images = []
    for src_type, src_path in test_sources.items():
        image_paths = sorted(list(find_images([src_path])))
        for i_path in image_paths:
            test_images.append(TestImage(i_path, src_type))
    print(f"Found {len(test_images):,} test images...")

    algorithms = build_algos()

    print("Launching worker processess...")
    with concurrent.futures.ProcessPoolExecutor() as executor:  # From https://stackoverflow.com/a/15143994/3915338
        print("Submitting images to workers...")
        futures = [executor.submit(test_an_image, image, algorithms) for image in test_images]
        for future in concurrent.futures.as_completed(futures):
            an_image = future.result()
            print(an_image.result_str(algorithms))
        print("Waiting ")
        concurrent.futures.wait(futures)

        # Un-hide exceptions
        exceptioned = [f for f in futures if f._exception]
        if exceptioned:
            raise exceptioned[0]._exception

        test_images = [f.result() for f in futures]


    print()

    if False:
        print(" = Disagreements = ")
        for test_image in test_images:
            sharps = map(lambda x: x == Sharpness.SHARP, test_image.actual_results.values())
            if any(sharps) and not all(sharps):
                print(test_image.result_str(algorithms))

    if True:
        print(" = Misses = ")
        for test_image in test_images:
            misses = any([x != test_image.expect_result for x in test_image.actual_results.values()])
            if misses:
                print(test_image.result_str(algorithms))

    if True:
        print(" = Double Misses = ")
        for test_image in test_images:
            misses = all([x != test_image.expect_result for x in test_image.actual_results.values()])
            if misses:
                print(test_image.result_str(algorithms))

    if True:
        print(" = Unanimous Blurries = ")
        for test_image in test_images:
            blurs = all([x == Sharpness.BLURRY for x in test_image.actual_results.values()])
            if blurs:
                print(test_image.result_str(algorithms))

    print(" = Algorithm results = ")
    for algo in algorithms.values():
        print(f" - {algo.name} - ")
        total = len(test_images)
        hits = len([ti for ti in test_images if ti.actual_results[algo.name] == ti.expect_result])
        misses = total - hits
        print(f"Hit rate: {hits * 100.0 / total:4.1f}% \tMiss rate: {misses * 100.0 / total:4.1f}%")

        all_scores = [ti.scores[algo.name] for ti in test_images]
        print("Scores = Min: {:>6.1f}\tMax: {:>6.1f}\tMean: {:>6.1f}".format(min(all_scores), max(all_scores),
                                                                       sum(all_scores) / len(all_scores)))

        all_times = [ti.times[algo.name] for ti in test_images]
        print(" Times = Min: {:>5.1f}s\tMax: {:>5.1f}s\tMean: {:>5.1f}s".format(min(all_times), max(all_times),
                                                                       sum(all_times) / len(all_times)))
        print()


if __name__ == '__main__':
    main()