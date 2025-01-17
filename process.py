import sys
import argparse
import logging
import pathlib
import json

import cv2
import rawpy
import numpy
from PIL import Image
from io import BytesIO

from blur_detection import estimate_blur, fix_image_size, pretty_blur_map
from blur_detection.detection import detect_blur_contours

from file_handling import find_images, load_image

red = "\033[0;31m"
green = "\033[0;32m"
white = "\033[0;37m"
yellow = "\033[0;33m"


def parse_args():
    parser = argparse.ArgumentParser(description='run blur detection on a single image')
    parser.add_argument('-i', '--images', type=str, nargs='+', required=True, help='directory of images')
    parser.add_argument('-s', '--save-path', type=str, default=None, help='path to save output')

    parser.add_argument('-tb', '--threshold_blur', type=float, default=90, help='blurry threshold')
    parser.add_argument('-ts', '--threshold_semi', type=float, default=115, help='semi-blurry threshold')
    parser.add_argument('-f', '--variable-size', action='store_true', help='fix the image size')

    parser.add_argument('-v', '--verbose', action='store_true', help='set logging level to debug')
    parser.add_argument('-d', '--display', action='store_true', help='display images')

    parser.add_argument('-m', '--move', action='store_true', help="move files based on result")

    return parser.parse_args()


def display_image(image_path, blur_map, descr):
    text = f"{image_path.name}: {descr}"

    blur_map = pretty_blur_map(blur_map)
    blur_map = cv2.cvtColor(blur_map, cv2.COLOR_GRAY2BGR)
    cv2.putText(blur_map, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow('result', blur_map)

    if cv2.waitKey(0) == ord('q'):
        logging.info('exiting...')
        exit()


def slope_calc(input_list: list[int]):
    indices = range(len(input_list))
    result = numpy.polyfit(indices, list(input_list), 1)
    slope = result[-2]
    return float(slope)


def show_summary(results):
    blurs, semis, sharps = [], [], []
    for r in results:
        if r["blurry"]:
            blurs.append(r)
        elif r["semi_blurry"]:
            semis.append(r)
        else:
            sharps.append(r)

    print(" = Summary = ")
    print(f"Blurry: {len(blurs):,} ({len(blurs)/len(results)*100.0:.1f}%)")
    print(f"Semi-blurry: {len(semis):,} ({len(semis) / len(results) * 100.0:.1f}%)")
    print(f"Sharp: {len(sharps):,} ({len(sharps) / len(results) * 100.0:.1f}%)")
    scores = [r["score"] for r in results]
    print(f"Blur scores - Min: {min(scores):,.1f} - Mean: {sum(scores)/len(scores):,.1f} - Max: {max(scores):,.1f}")


if __name__ == '__main__':
    if not sys.version_info >= (3, 6):
        raise Exception("Requires at least Python 3.6. Found: ", sys.version_info)
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    fix_size = not args.variable_size
    logging.info(f'fix_size: {fix_size}')

    if args.save_path is not None:
        save_path = pathlib.Path(args.save_path)
        assert save_path.suffix == '.json', save_path.suffix
    else:
        save_path = None

    results = []

    for image_path in find_images(args.images):
        print()
        if any(r["input_path"].lower() == str(image_path).lower() for r in results):
            logging.debug(f"Skipping {image_path} because it was already processed. Probably related to case.")
            continue

        image = load_image(str(image_path))
        if image is None:
            logging.warning(f'warning! failed to read image from {image_path}; skipping!')
            continue

        #logging.info(f'processing {image_path}')

        if fix_size:
            image = fix_image_size(image)
        else:
            logging.warning('not normalizing image size for consistent scoring!')

        score = detect_blur_contours(image)

        blurry = bool(score < args.threshold_blur)
        semi_blurry = bool(args.threshold_blur <= score < args.threshold_semi) # TODO: Numbered bins?
        descr = "Blurry" if blurry else "Semi-Blurry" if semi_blurry else "Sharp"
        color = red if blurry else yellow if semi_blurry else white

        print(color, f'image_path: {image_path.name:12} - score: {score:5,.1f} ({descr})', white)
        results.append({'input_path': str(image_path), 'score': score, 'blurry': blurry, 'semi_blurry': semi_blurry})

        if args.display:
            display_image(image_path, blur_map, descr)

        if args.move:
            blur_folder = image_path.parent/"blurry"
            blur_folder.mkdir(exist_ok=True)
            semi_blur_folder = image_path.parent/"semi_blurry"
            semi_blur_folder.mkdir(exist_ok=True)

            dest_folder = blur_folder if blurry else semi_blur_folder if semi_blurry else None
            if dest_folder:
                new_path = dest_folder/image_path.name
                image_path.replace(new_path)
                logging.debug(f"Moved {image_path.name} into the {dest_folder}...")
            else:
                logging.debug(f"Not moving {image_path} as it's not blurry")

    if len(results) == 0:
        print("No images found with specified input paths. Exiting...")
        exit(1)

    show_summary(results)

    if save_path is not None:
        logging.info(f'saving json to {save_path}')

        with open(save_path, 'w') as result_file:
            data = {'images': args.images, 'threshold': args.threshold_blur,
                    'threshold_semi': args.threshold_semi, 'fix_size': fix_size, 'results': results}
            json.dump(data, result_file, indent=4)
