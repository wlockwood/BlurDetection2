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

from blur_detection import estimate_blur
from blur_detection import fix_image_size
from blur_detection import pretty_blur_map

raw_types = ['.nef', '.cr2', '.cr3']
normal_types = ['.jpg', '.jpeg', '.png']

red = "\033[0;31m"
green = "\033[0;32m"
white = "\033[0;37m"
yellow = "\033[0;33m"


def parse_args():
    parser = argparse.ArgumentParser(description='run blur detection on a single image')
    parser.add_argument('-i', '--images', type=str, nargs='+', required=True, help='directory of images')
    parser.add_argument('-s', '--save-path', type=str, default=None, help='path to save output')

    parser.add_argument('-tb', '--threshold_blur', type=float, default=15.0, help='blurry threshold')
    parser.add_argument('-ts', '--threshold_semi', type=float, default=25.0, help='semi-blurry threshold')
    parser.add_argument('-f', '--variable-size', action='store_true', help='fix the image size')

    parser.add_argument('-v', '--verbose', action='store_true', help='set logging level to debug')
    parser.add_argument('-d', '--display', action='store_true', help='display images')

    parser.add_argument('-m', '--move', action='store_true', help="move files based on result")

    return parser.parse_args()


def find_images(image_paths, img_extensions=raw_types + normal_types):
    img_extensions += [i.upper() for i in img_extensions]

    for path in image_paths:
        path = pathlib.Path(path)

        if path.is_file():
            logging.info(f"Found image file '{path}'")
            if path.suffix not in img_extensions:
                logging.info(f'{path.suffix} is not an image extension! skipping {path}')
                continue
            else:
                yield path

        if path.is_dir():
            logging.info(f"Found directory '{path}'")
            for img_ext in img_extensions:
                yield from path.glob(f'*{img_ext}')


def load_image(path):
    is_raw = any(e in path.lower() for e in raw_types)
    if is_raw:
        with rawpy.imread(path) as raw:
            # raises rawpy.LibRawNoThumbnailError if thumbnail missing
            # raises rawpy.LibRawUnsupportedThumbnailError if unsupported format
            thumb = raw.extract_thumb()

        if thumb.format == rawpy.ThumbFormat.JPEG:
            # thumb.data is already in JPEG format, save as-is
            bio = BytesIO(thumb.data)
            pilmage = Image.open(bio)
            out = cv2.cvtColor(numpy.array(pilmage), cv2.COLOR_RGB2BGR)

        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            #cv2.cvtColor(numpy.array(pilmage), cv2.COLOR_RGB2BGR)
            out = thumb.data
    else:
        out = cv2.imread(str(path))
    if out is None:
        raise Exception(f"Failed to load image data from {path}")
    return out


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

        image = cv2.bilateralFilter(image, 5, 75, 75)  # Blur prior to downsampling?
        blur_map, score = estimate_blur(image)
        blurry = bool(score < args.threshold_blur)
        semi_blurry = bool(args.threshold_blur <= score < args.threshold_semi)
        descr = "Blurry" if blurry else "Semi-Blurry" if semi_blurry else "Sharp"
        color = red if blurry else yellow if semi_blurry else white

        print(color, f'image_path: {image_path.name:12} - score: {score:5,.1f} ({descr})', white)
        results.append({'input_path': str(image_path), 'score': score, 'blurry': blurry, 'semi_blurry': semi_blurry})

        if args.display:

            text = f"{image_path.name}: {descr}"

            blur_map = pretty_blur_map(blur_map)
            blur_map = cv2.cvtColor(blur_map, cv2.COLOR_GRAY2BGR)
            cv2.putText(blur_map, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            cv2.imshow('result', blur_map)

            if cv2.waitKey(0) == ord('q'):
                logging.info('exiting...')
                exit()

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

    print(" = Summary = ")
    blurs, semis, sharps = [], [], []
    for r in results:
        if r["blurry"]:
            blurs.append(r)
        elif r["semi_blurry"]:
            semis.append(r)
        else:
            sharps.append(r)

    print(f"Blurry: {len(blurs):,} ({len(blurs)/len(results)*100.0:.1f}%)")
    print(f"Semi-blurry: {len(semis):,} ({len(semis) / len(results) * 100.0:.1f}%)")
    print(f"Sharp: {len(sharps):,} ({len(sharps) / len(results) * 100.0:.1f}%)")
    scores = [r["score"] for r in results]
    print(f"Blur scores - Min: {min(scores):,.1f} - Mean: {sum(scores)/len(scores):,.1f} - Max: {max(scores):,.1f}")

    if save_path is not None:
        logging.info(f'saving json to {save_path}')

        with open(save_path, 'w') as result_file:
            data = {'images': args.images, 'threshold': args.threshold_blur,
                    'threshold_semi': args.threshold_semi, 'fix_size': fix_size, 'results': results}
            json.dump(data, result_file, indent=4)
