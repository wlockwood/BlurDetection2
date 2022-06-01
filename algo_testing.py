from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from enum import Enum, auto
from typing import Callable
import cv2

from time import perf_counter as pc

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
        return f"{test_image.path.name:>20} {test_image.expect_result.name:6} --- {self.scores_str(algorithms)}"


test_sources = {
    Sharpness.SHARP: r"C:\Users\jinks\Desktop\Export\blurdet\sharp",
    Sharpness.BLURRY: r"C:\Users\jinks\Desktop\Export\blurdet\blurry"
}

image_extensions = ['.jpg', '.jpeg', '.png']

test_images = []
for src_type, src_path in test_sources.items():
    image_paths = sorted(list(find_images([src_path])))
    for i_path in image_paths:
        test_images.append(TestImage(i_path, src_type))
print(f"Found {len(test_images):,} test images...")

algorithms = {
    "Laplace": Algorithm("Laplace", lambda i: estimate_blur(i, just_score=True), 10),
    "FFT": Algorithm("FFT", detect_blur_fft, 20),
}

for test_image in test_images:
    image = load_image(test_image.path)
    image = fix_image_size(image, 2e6)

    for algo in algorithms.values():
        t0 = pc()
        score = algo.call_func(image)
        elapsed = pc() - t0

        test_image.scores[algo.name] = score
        test_image.times[algo.name] = elapsed

        sharp: bool = score >= algorithms[algo.name].blurriness_threshold
        sharpness = Sharpness.SHARP if sharp else Sharpness.BLURRY
        test_image.actual_results[algo.name] = sharpness
    score_str = test_image.scores_str(algorithms)

    # Suppress hits
    # miss = any(x != test_image.expect_result for x in test_image.actual_results.values())
    # if miss:
    print(f"{test_image.path.name:>20} {test_image.expect_result.name:6} --- {score_str}")

print()
print(" = Algorithm results = ")
for algo in algorithms.values():
    print(f" - {algo.name} - ")
    total = len(test_images)
    hits = len([ti for ti in test_images if ti.actual_results[algo.name] == ti.expect_result])
    misses = total - hits
    print(f"Hit rate: {hits * 100.0 / total:4.1f}% \tMiss rate: {misses * 100.0 / total:4.1f}%")

    all_scores = [ti.scores[algo.name] for ti in test_images]
    print("Scores = Min: {:.1f}\tMax: {:.1f}\tMean: {:.1f}".format(min(all_scores), max(all_scores),
                                                                   sum(all_scores) / len(all_scores)))

    all_times = [ti.times[algo.name] for ti in test_images]
    print(" Times = Min: {:.1f}\tMax: {:.1f}\tMean: {:.1f}".format(min(all_times), max(all_times),
                                                                   sum(all_times) / len(all_times)))
    print()

print(" = Disagreements = ")
for test_image in test_images:
    sharps = map(lambda x: x == Sharpness.SHARP, test_image.actual_results.values())
    if any(sharps) and not all(sharps):
        print(test_image.result_str(algorithms))
